import time
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from gradio import ChatMessage


class PDChatbot:
    PROMPT_TEMPLATE = """You are a helpful clinical decision support AI for psychiatric disorder diagnosis. Always:
    1. Think step-by-step before responding
    2. Justify your initial assessment and the interpretation of HRV metrics, referencing clinical guidelines or evidence when possible.
    3. When the finalization request is queried, you must finalize the decision (only answer "healthy" or "treatment") but you may overturn your prior assessment if, after reviewing all evidence, you are confident a different answer is correct. Clearly state the reason for any change.
    4. Provide accurate, current information using clinical guidelines
    5. Avoid assumptions. Only use the provided data
    6. Cross-validate findings with multiple sources
    7. Flag urgent concerns immediately
    8. Reference sources for non-standard conclusions
    9. Maintain clarity with very concise and straightforward responses"""

    def __init__(
        self, model_name, api_key, temperature=0.2, top_p=0.7, max_tokens=4096
    ):
        self.client = ChatNVIDIA(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def generate_response(self, history, new_message):
        # First, add the user message to the history
        history.append({"role": "user", "content": new_message})

        # Then build the messages list for the API call
        messages = [{"role": "system", "content": self.PROMPT_TEMPLATE}]
        for msg in history[
            :-1
        ]:  # Exclude the just-added user message since we'll add it below
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Create an empty assistant message for streaming
        thinking_msg = self._create_thinking_message()
        history.append(thinking_msg)

        # Stream the response
        for chunk in self.client.stream(
            messages + [{"role": "user", "content": new_message}]
        ):
            history[-1]["content"] += chunk.content
            yield history

    def _create_thinking_message(self):
        return {
            "role": "assistant",
            "content": "",
        }
