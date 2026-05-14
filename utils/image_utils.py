from PIL import Image
import io

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}
MAX_DIMENSION = 1024
MAX_SIZE_MB = 4


def validate_and_process(uploaded_file) -> tuple[bytes, str]:
    """
    Validate file type, resize if needed, return (image_bytes, error_message).
    Returns (b"", error_message) on failure.
    """
    raw_bytes = uploaded_file.read()

    if len(raw_bytes) > MAX_SIZE_MB * 1024 * 1024:
        return b"", f"File too large. Maximum size is {MAX_SIZE_MB} MB."

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        # Validate against actual file content, not the browser-supplied MIME type
        if img.format not in ALLOWED_FORMATS:
            return b"", f"Unsupported image format: {img.format}. Please upload JPEG, PNG, WebP, or GIF."

        # Convert RGBA or P mode to RGB for JPEG compatibility
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        # Resize if too large
        if max(img.size) > MAX_DIMENSION:
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
