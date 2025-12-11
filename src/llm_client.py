import requests
from src.config import CONFIG

class LLMClient:
    """
    Handles communication with Groq Cloud API for intelligence.
    """
    def __init__(self):
        self.api_key = CONFIG.GROQ_API_KEY
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile" # Updated from decommissioned model

    def process_text(self, glosses: list[str]) -> str:
        """
        Refines glosses into a sentence using Llama-3.
        """
        raw_text = " ".join(glosses)
        prompt = f"""
        You are an expert ASL (Sign Language) interpreter.
        Convert the raw glosses into a natural, grammatically correct English sentence.
        
        Rules:
        1. Vocabulary: Treat "MINEMY" always as "MY". Treat "ME" as "I" or "ME" based on context.
        2. Output: Provide ONLY the final sentence. NO explanations, NO quotes, NO emotional descriptors.
        
        Examples:
        Input: ME HUNGRY
        Output: I am hungry.
        Input: MINEMY UNCLE
        Output: My uncle.
        Input: MINEMY HORSE SEE
        Output: I see my horse.
        
        Input: {raw_text}
        Output:
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 64
        }
        
        try:
            print(f"Sending to Groq ({self.model})...")
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            result = data['choices'][0]['message']['content'].strip()
            # Clean quotes if any
            result = result.replace('"', '').replace("'", "")
            # print(f"[INFO] Translation result: {result}")
            return result
            
        except Exception as e:
            print(f"[ERROR] External API request failed: {e}")
            return raw_text.capitalize() # Fallback
