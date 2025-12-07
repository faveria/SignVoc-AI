from .utils import Preprocess
from .config import MAX_LEN, CHANNELS, NUM_CLASSES
import tensorflow as tf
from typing import Optional, Callable, Dict, Any, Union, Tuple, List

class ECA(tf.keras.layers.Layer):
    """
    Efficient Channel Attention layer.

    Args:
        kernel_size (int): Size of the kernel for the convolutional layer.
    """

    def __init__(self, kernel_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Applies the efficient channel attention mechanism.
        """
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn

class LateDropout(tf.keras.layers.Layer):
    """
    Layer that applies dropout after a certain training step.

    Args:
        rate (float): Dropout rate.
        noise_shape: Shape of the binary dropout mask.
        start_step (int): The training step after which the dropout is applied.
    """
    def __init__(self, rate: float, noise_shape: Optional[Any] = None, start_step: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)

    def build(self, input_shape):
        super().build(input_shape)
        # Using add_weight instead of self._train_counter to be more Keras-compliant
        self.train_counter = self.add_weight(
            name="train_counter",
            shape=(),
            dtype="int64",
            initializer="zeros",
            trainable=False
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Applies dropout to the input tensor conditionally.
        """
        x = tf.cond(
            self.train_counter < self.start_step, 
            lambda: inputs, 
            lambda: self.dropout(inputs, training=training)
        )
        if training:
            self.train_counter.assign_add(1)
        return x

class CausalDWConv1D(tf.keras.layers.Layer):
    """
    Causal Dilated Depthwise Convolutional 1D layer.

    Args:
        kernel_size (int): Size of the kernel.
        dilation_rate (int): Dilation rate.
        use_bias (bool): Whether to use bias.
        depthwise_initializer: Initializer for the depthwise kernel.
    """
    
    def __init__(self, 
        kernel_size: int = 17,
        dilation_rate: int = 1,
        use_bias: bool = False,
        depthwise_initializer: str = 'glorot_uniform',
        name: str = '', **kwargs):
        super().__init__(name=name, **kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0), name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
                            kernel_size,
                            strides=1,
                            dilation_rate=dilation_rate,
                            padding='valid',
                            use_bias=use_bias,
                            depthwise_initializer=depthwise_initializer,
                            name=name + '_dwconv')
        self.supports_masking = True
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x

def Conv1DBlock(channel_size: int,
                kernel_size: int,
                dilation_rate: int = 1,
                drop_rate: float = 0.0,
                expand_ratio: int = 2,
                se_ratio: float = 0.25,
                activation: str = 'swish',
                name: Optional[str] = None) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Efficient Conv1D block.
    """

    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))

    def apply(inputs: tf.Tensor) -> tf.Tensor:
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(
            channels_expand,
            use_bias=True,
            activation=activation,
            name=name + '_expand_conv')(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(kernel_size,
            dilation_rate=dilation_rate,
            use_bias=False,
            name=name + '_dwconv')(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

        x  = ECA()(x)

        x = tf.keras.layers.Dense(
            channel_size,
            use_bias=True,
            name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + '_drop')(x)

        if (channels_in == channel_size):
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x

    return apply

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    Multi-Head Self-Attention layer.
    """
    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        qkv = self.qkv(inputs)
        # Reshape to (Batch, Time, Heads, Dim_per_head)
        qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
             # Mask should be broadcastable
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x


def TransformerBlock(dim: int = 256, num_heads: int = 4, expand: int = 4, attn_dropout: float = 0.2, drop_rate: float = 0.2, activation: str = 'swish') -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Transformer Block.
    """
    def apply(inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x
    return apply

class TFLiteModel(tf.Module):
    """
    TensorFlow Lite model that takes input tensors and applies:
        – A Preprocessing Model
        – The ISLR model 
    """

    def __init__(self, islr_models: List[tf.keras.Model]):
        super(TFLiteModel, self).__init__()
        self.prep_inputs = Preprocess()
        self.islr_models = islr_models
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        x = self.prep_inputs(tf.cast(inputs, dtype=tf.float32))
        # Loop over models and average outputs
        outputs = [model(x) for model in self.islr_models]
        if len(outputs) > 1:
            outputs = tf.keras.layers.Average()(outputs)
        else:
            outputs = outputs[0]
            
        return {'outputs': outputs}

def get_model(max_len: int = MAX_LEN, dropout_step: int = 0, dim: int = 192) -> tf.keras.Model:
    """
    Creates the ISLR model.
    """
    inp = tf.keras.Input((max_len, CHANNELS))
    # x = tf.keras.layers.Masking(mask_value=PAD, input_shape=(max_len, CHANNELS))(inp) 
    x = inp
    ksize = 17
    
    # Stem layers
    x = tf.keras.layers.Dense(dim, use_bias=False, name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name='stem_bn')(x)

    # Convolutional and Transformer blocks
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)

    # Additional blocks for larger models
    if dim == 384: 
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = TransformerBlock(dim, expand=2)(x)

        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = TransformerBlock(dim, expand=2)(x)

    # Top layers
    x = tf.keras.layers.Dense(dim * 2, activation=None, name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, name='classifier', activation='softmax')(x)
    return tf.keras.Model(inp, x)
