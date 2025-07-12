import requests
import base64
from PIL import Image
from typing import Dict, Any
from config import OPENROUTER_API_KEY, Path

class QwenClient:
    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost", 
            "Content-Type": "application/json"
        }

    def generate_image_caption(self, image_path: str) -> str:
        """Use Qwen2.5-VL (via OpenRouter) to caption an image."""
        try:
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
                image_base64 = base64.b64encode(image_bytes).decode()

            payload = {
                "model": "qwen/qwen2.5-vl-32b-instruct:free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Describe this image in detail. Focus on visible objects, structure, labels, text, and any spatial patterns. "
                                    "Be helpful and accurate — the output will be used as part of a document understanding pipeline."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ]
            }

            response = requests.post(self.api_url, headers=self.headers, json=payload)
            result = response.json()

            if "choices" in result:
                caption = result["choices"][0]["message"]["content"]
                if not caption.strip():
                    print(f"❌ Empty caption returned by Qwen for image: {image_path}")
                    return "[⚠️ Caption failed]"
                return caption
            else:
                print(f"❌ Qwen API error response: {result}")
                return "[⚠️ Caption error: No content]"
        except Exception as e:
            print(f"❌ Exception in Qwen captioning: {e}")
            return f"[⚠️ Caption error: {Path(image_path).name}]"
