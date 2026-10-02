"""Small real attachments used by multimodal regression tests."""

from io import BytesIO

from PIL import Image


def png_bytes(color: str = "red", size: tuple[int, int] = (32, 24)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def pdf_bytes(pages: int = 2) -> bytes:
    output = BytesIO()
    images = [
        Image.new("RGB", (100, 140), "red" if index % 2 == 0 else "blue")
        for index in range(pages)
    ]
    images[0].save(output, format="PDF", save_all=True, append_images=images[1:])
    return output.getvalue()
