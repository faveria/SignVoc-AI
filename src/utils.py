from .config import MAX_LEN, POINT_LANDMARKS
import tensorflow as tf
from typing import Optional, List, Union

def tf_nan_mean(x: tf.Tensor, axis: Union[int, List[int]] = 0, keepdims: bool = False) -> tf.Tensor:
    """
    Computes the mean of the input tensor while ignoring NaN values.

    Args:
        x (tf.Tensor): Input tensor.
        axis (Union[int, List[int]], optional): Axis along which to compute the mean. Defaults to 0.
        keepdims (bool, optional): Whether to keep the dimensions of the input tensor. Defaults to False.

    Returns:
        tf.Tensor: The mean of the input tensor with NaN values ignored.
    """
    # Replace NaN with 0 and count non-NaN values
    mask = tf.math.is_nan(x)
    x_zero_imputed = tf.where(mask, tf.zeros_like(x), x)
    non_nan_count = tf.where(mask, tf.zeros_like(x), tf.ones_like(x))
    
    sum_val = tf.reduce_sum(x_zero_imputed, axis=axis, keepdims=keepdims)
    count_val = tf.reduce_sum(non_nan_count, axis=axis, keepdims=keepdims)
    
    # Avoid division by zero
    return tf.math.divide_no_nan(sum_val, count_val)

def tf_nan_std(x: tf.Tensor, center: Optional[tf.Tensor] = None, axis: Union[int, List[int]] = 0, keepdims: bool = False) -> tf.Tensor:
    """
    Computes the standard deviation of the input tensor while ignoring NaN values.

    Args:
        x (tf.Tensor): Input tensor.
        center (Optional[tf.Tensor], optional): Tensor representing the mean. If None, computed internally. Defaults to None.
        axis (Union[int, List[int]], optional): Axis along which to compute the std. Defaults to 0.
        keepdims (bool, optional): Whether to keep dimensions. Defaults to False.

    Returns:
        tf.Tensor: The standard deviation of the input tensor with NaN values ignored.
    """
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

class Preprocess(tf.keras.layers.Layer):
    """
    Preprocessing layer for input data.

    Args:
        max_len (int): Maximum length of the input sequence. Default is MAX_LEN from config.
        point_landmarks (List[int]): List of point landmarks to extract. Default is POINT_LANDMARKS.
    """
    
    def __init__(self, max_len: int = MAX_LEN, point_landmarks: List[int] = POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Preprocesses the input data.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Preprocessed tensor.
        """
        
        def expand_dims():
            return inputs[None, ...]
            
        def identity():
            return inputs
            
        x = tf.cond(tf.equal(tf.rank(inputs), 3), expand_dims, identity)
        
        # Normalize around the face center (landmark 17 is typically nose/face center in some meshes, or top lip)
        # Assuming index 17 based on original config logic.
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        
        x = tf.gather(x, self.point_landmarks, axis=2) # N,T,P,C
        std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
        
        x = (x - mean) / std

        if self.max_len is not None:
            x = x[:, :self.max_len]
        
        length = tf.shape(x)[1]
        x = x[..., :2] # Only use X and Y coordinates, drop Z if present

        dx = tf.cond(
            tf.shape(x)[1] > 1,
            lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]),
            lambda: tf.zeros_like(x)
        )

        dx2 = tf.cond(
            tf.shape(x)[1] > 2,
            lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]),
            lambda: tf.zeros_like(x)
        )

        x = tf.concat([
            tf.reshape(x, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx2, (-1, length, 2 * len(self.point_landmarks))),
        ], axis=-1)
        
        x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
        
        return x
