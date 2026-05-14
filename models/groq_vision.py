import base64
from typing import Generator

from groq import Groq

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_SYSTEM_ANALYST = (
    "You are an expert visual analyst. Analyze images thoroughly and accurately. "
    "Cover: main subjects, colors, composition, context, text if present, "
    "and any notable details."
)

_SYSTEM_QA = (
    "You are a precise visual analyst answering questions about an uploaded image. "
    "Be specific, factual, and concise. If something is not visible in the image, "
    "say so clearly."
)


def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _build_qa_messages(data_url: str, question: str, conversation_history: list) -> list:
    """Build the messages list for Q&A, correctly embedding the image in the first user turn."""
    messages = [{"role": "system", "content": _SYSTEM_QA}]

    if not conversation_history:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": question},
            ],
        })
    else:
        # Embed the image inside the first history user message so we never produce
        # two consecutive user-role messages (which some model endpoints reject).
        for i, msg in enumerate(conversation_history):
            if i == 0 and msg["role"] == "user":
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": msg["content"]},
                    ],
                })
            else:
                messages.append(msg)
        messages.append({"role": "user", "content": question})

    return messages


def get_initial_analysis(image_bytes: bytes, api_key: str) -> tuple[str, int]:
    """Deep initial analysis of the image. Returns (analysis_text, tokens_used)."""
    try:
        client = Groq(api_key=api_key)
        data_url = f"data:image/jpeg;base64,{encode_image(image_bytes)}"

        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_ANALYST},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {
                            "type": "text",
                            "text": (
                                "Provide a comprehensive analysis of this image. Cover: "
                                "(1) What is shown, (2) Key details and objects, "
                                "(3) Colors and composition, (4) Any text visible, "
                                "(5) Context or likely use case of this image."
                            ),
                        },
                    ],
                },
            ],
            max_tokens=600,
        )

        text = response.choices[0].message.content or ""
        usage = response.usage
        tokens = int(usage.total_tokens) if usage and usage.total_tokens is not None else 0
        return text, tokens
    except Exception as e:
        return f"Groq API error: {str(e)}", 0


def answer_followup(
    image_bytes: bytes,
    question: str,
    conversation_history: list,
    api_key: str,
) -> tuple[str, int]:
    """Non-streaming Q&A. Returns (answer_text, tokens_used)."""
    try:
        client = Groq(api_key=api_key)
        data_url = f"data:image/jpeg;base64,{encode_image(image_bytes)}"
        messages = _build_qa_messages(data_url, question, conversation_history)

        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=messages,
            max_tokens=400,
        )

        text = response.choices[0].message.content or ""
        usage = response.usage
        tokens = int(usage.total_tokens) if usage and usage.total_tokens is not None else 0
        return text, tokens
    except Exception as e:
        return f"Groq API error: {str(e)}", 0


def answer_followup_stream(
    image_bytes: bytes,
    question: str,
    conversation_history: list,
    api_key: str,
    token_bucket: list,
) -> Generator[str, None, None]:
    """Streaming Q&A — yields text chunks; appends total_tokens to token_bucket when done."""
    try:
        client = Groq(api_key=api_key)
        data_url = f"data:image/jpeg;base64,{encode_image(image_bytes)}"
        messages = _build_qa_messages(data_url, question, conversation_history)

        stream = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=messages,
            max_tokens=400,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if getattr(chunk, "usage", None):
                token_bucket.append(chunk.usage.total_tokens)

    except Exception as e:
        yield f"Groq API error: {str(e)}"


def get_suggested_questions(analysis_text: str) -> list[str]:
    """Generate context-aware follow-up suggestions from the analysis (no extra API call)."""
    lower = analysis_text.lower()
    suggestions: list[str] = []

    if any(w in lower for w in ["text", "word", "sign", "label", "written", "printed", "caption"]):
        suggestions.append("Can you read all the text visible in this image?")
    if any(w in lower for w in ["person", "people", "man", "woman", "child", "face", "figure", "human"]):
        suggestions.append("Describe the people in this image in detail")
    if any(w in lower for w in ["chart", "graph", "table", "data", "diagram", "plot", "bar", "pie"]):
        suggestions.append("Summarize the data or information shown")
    if any(w in lower for w in ["product", "brand", "logo", "package", "item"]):
        suggestions.append("Write a product description for this item")
    if any(w in lower for w in ["building", "architecture", "room", "indoor", "outdoor", "street", "landscape", "scene"]):
        suggestions.append("Describe the setting or environment in detail")
    if any(w in lower for w in ["animal", "dog", "cat", "bird", "wildlife", "creature"]):
        suggestions.append("What animal is this and what can you tell about it?")

    defaults = [
        "List all objects you can identify",
        "What is the dominant color palette?",
        "What is the mood or atmosphere of this image?",
        "What context or setting does this come from?",
    ]
    for d in defaults:
        if len(suggestions) < 4:
            suggestions.append(d)

    return suggestions[:4]
