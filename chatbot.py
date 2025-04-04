import time
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from gradio import ChatMessage

# client = ChatNVIDIA(
#     model="meta/llama-3.3-70b-instruct",
#     api_key="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",
#     temperature=0.2,
#     top_p=0.7,
#     max_tokens=4096,
# )


# def generate_response(history, new_message):
#     # Append user message to the existing history
#     updated_history = history + [{"role": "user", "content": new_message}]
#     # Add an empty assistant message to start accumulating the response
#     updated_history.append({"role": "assistant", "content": ""})

#     # Prepare the messages for the model (exclude the empty assistant message)
#     messages_for_model = updated_history[:-1]

#     # Stream the response from the model
#     for chunk in client.stream(messages_for_model):
#         updated_history[-1]["content"] += chunk.content  # Accumulate chunks
#         yield updated_history  # Yield updated history with incremental response

#     return updated_history  # Return final history after streaming completes


class PDChatbot:
    PROMPT_TEMPLATE = """You are a helpful AI assistant in psychiatric disorder diagnosis. Always:
    1. Think step-by-step before responding
    2. Validate information from multiple sources
    3. Provide accurate and up-to-date information
    4. Avoid making assumptions
    5. Be empathetic and understanding
    6. Respect user privacy and confidentiality
    7. Provide resources for further information
    8. Report any concerning or harmful content
    9. Avoid providing medical advice
    10. Cite references when applicable
    11. Be concise and clear in your responses"""

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
