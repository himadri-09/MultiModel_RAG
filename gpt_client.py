from openai import AzureOpenAI
from typing import List, Dict, Any
from config import AZURE_OPENAI_API_KEY,AZURE_OPENAI_ENDPOINT,AZURE_DEPLOYMENT_NAME,AZURE_API_VERSION


class GPTClient:
    """Answer generator that calls your Azure‑hosted GPT deployment."""
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )
        self.deployment_name = AZURE_DEPLOYMENT_NAME

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        try:
            if not context_chunks:
                final_prompt = query
            else:
                context_parts = []
                for i, c in enumerate(context_chunks):
                    page = c.get("page_number", "N/A")
                    if c["type"] == "text":
                        context_parts.append(f"Text {i+1}:\n{c['content']}")
                    elif c["type"] == "table":
                        context_parts.append(f"Table {i+1} (p.{page}):\n{c['content']}")
                    elif c["type"] == "image":
                        context_parts.append(f"Image {i+1} (p.{page}):\n{c['content']}")
                context_text = "\n\n".join(context_parts)
                final_prompt = f"""
Based on the following Vastu context, answer the user's question.

Context:
{context_text}

Question:
{query}

Instructions:
- Answer **only** from the context.
- Cite rules or page numbers when relevant.
- If info is missing, say so.
""".strip()

            resp = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful Vastu assistant."},
                    {"role": "user", "content": final_prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Error generating answer: {e}"
