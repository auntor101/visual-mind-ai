import time

import requests

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
# Primary: new HuggingFace router. Fallback: classic free-tier endpoint.
BLIP_API_URL_NEW = f"https://router.huggingface.co/hf-inference/models/{BLIP_MODEL_ID}"
BLIP_API_URL_OLD = f"https://api-inference.huggingface.co/models/{BLIP_MODEL_ID}"


def _parse_caption(result) -> str:
    if isinstance(result, list) and result:
        return result[0].get("generated_text", "No caption generated")
    if isinstance(result, dict) and "generated_text" in result:
        return str(result["generated_text"])
    return "No caption generated"


def _post_with_retry(url: str, headers: dict, image_bytes: bytes, retries: int = 3) -> str | None:
    """
    POST to a HuggingFace inference URL.
    Returns a caption string on success, None if this URL should not be used,
    or an error string starting with a warning emoji.
    """
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=image_bytes, timeout=60)

            if response.status_code == 200:
                return _parse_caption(response.json())

            if response.status_code == 503:
                # Model is cold — wait and retry
                try:
                    wait = min(response.json().get("estimated_time", 20), 30)
                except Exception:
                    wait = 20
                if attempt < retries - 1:
                    time.sleep(wait)
                    continue
                return f"⏳ Model still warming up (~{int(wait)}s). Try again in a moment."

            if response.status_code == 429:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return "⚠️ Rate limit hit. Wait a moment and try again."

            if response.status_code == 401:
                return "❌ Invalid HuggingFace token. Check it at huggingface.co/settings/tokens"

            if response.status_code in (403, 404):
                # Router doesn't support this model — signal caller to try fallback
                return None

            return f"API error {response.status_code}: {response.text[:150]}"

        except requests.Timeout:
            if attempt < retries - 1:
                continue
            return "⏱️ Request timed out. HuggingFace may be slow — try again."
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    return "Caption unavailable after retries."


def get_blip_caption(image_bytes: bytes, hf_token: str) -> str:
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "image/jpeg",
    }

    # Try classic endpoint first (stable); fall back to new router if it signals None
    result = _post_with_retry(BLIP_API_URL_OLD, headers, image_bytes)
    if result is not None:
        return result

    return _post_with_retry(BLIP_API_URL_NEW, headers, image_bytes) or "Caption unavailable."
