import base64

from groq import Groq

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def get_initial_analysis(image_bytes: bytes, api_key: str) -> tuple[str, int]:
    """
    Perform deep initial analysis of the image using Groq Llama 4 Scout (vision).
    Returns (analysis_text, tokens_used).
    """
    try:
        client = Groq(api_key=api_key)
        base64_image = encode_image(image_bytes)
        data_url = f"data:image/jpeg;base64,{base64_image}"

        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert visual analyst. Analyze images thoroughly and accurately. "
                        "Cover: main subjects, colors, composition, context, text if present, "
                        "and any notable details."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
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
    """
    Answer a follow-up question about the image, maintaining conversation context.
    Returns (answer_text, tokens_used).
    """
    try:
        client = Groq(api_key=api_key)
        base64_image = encode_image(image_bytes)
        data_url = f"data:image/jpeg;base64,{base64_image}"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise visual analyst answering questions about an uploaded image. "
                    "Be specific, factual, and concise. If something is not visible in the image, "
                    "say so clearly."
                ),
            }
        ]

        if not conversation_history:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {"type": "text", "text": question},
                    ],
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {"type": "text", "text": "Referring to the image above:"},
                    ],
                }
            )
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": question})

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
