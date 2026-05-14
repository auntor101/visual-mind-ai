from PIL import Image
import io

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_DIMENSION = 1024
MAX_SIZE_MB = 4


def validate_and_process(uploaded_file) -> tuple[bytes, str]:
    """
    Validate file type, resize if needed, return (image_bytes, error_message).
    Returns ("", error_message) on failure.
    """
    if uploaded_file.type not in ALLOWED_TYPES:
        return b"", f"Unsupported file type: {uploaded_file.type}. Please upload JPEG, PNG, WebP, or GIF."

    raw_bytes = uploaded_file.read()

    try:
        img = Image.open(io.BytesIO(raw_bytes))

        # Convert RGBA or P mode to RGB for JPEG compatibility
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        # Resize if too large
        if max(img.size) > MAX_DIMENSION or len(raw_bytes) > MAX_SIZE_MB * 1024 * 1024:
            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return buffer.getvalue(), ""

        # Re-encode as JPEG for consistency
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue(), ""

    except Exception as e:
        return b"", f"Could not process image: {str(e)}"


def get_image_info(image_bytes: bytes) -> dict:
    """Return basic image metadata."""
    img = Image.open(io.BytesIO(image_bytes))
    size_kb = len(image_bytes) / 1024
    return {
        "width": img.size[0],
        "height": img.size[1],
        "mode": img.mode,
        "size_kb": round(size_kb, 1),
    }
