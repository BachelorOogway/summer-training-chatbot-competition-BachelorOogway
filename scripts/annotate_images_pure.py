#!/usr/bin/env python3
"""Annotate images in a folder using Azure OpenAI GPT-4o-mini vision API.

Images are analyzed and annotations are saved to a JSON file.

Usage:
    python scripts/annotate_images.py [image_folder] [output_file]

Defaults:
    image_folder: images_test/
    output_file: image_annotations.json
"""
import sys
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
import os
from openai import AzureOpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_Key = os.getenv("AZURE_OPENAI_API_KEY")
if not API_Key:
    raise RuntimeError("Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in .env or environment.")

# Initialize GPT client with vision capability
gpt_client = AzureOpenAI(
    azure_endpoint="https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
    api_key=API_Key,
    api_version="2025-01-01-preview",
)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_media_type(image_path: Path) -> str:
    """Get MIME type from file extension."""
    ext = image_path.suffix.lower()
    type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return type_map.get(ext, "image/jpeg")


def annotate_image(image_path: Path) -> str:
    """Send image to GPT-4o-mini and get annotation."""
    try:
        image_data = encode_image(image_path)
        media_type = get_media_type(image_path)
        
        response = gpt_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please provide a detailed annotation/description of this image. Include: what is shown, key objects, colors, layout, and any text visible.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing image: {str(e)}"


def main():
    image_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("images_test")
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("image_annotations.json")

    if not image_folder.exists():
        print(f"Image folder not found: {image_folder}")
        sys.exit(1)

    # Get all supported image files
    image_files = [
        f for f in image_folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not image_files:
        print(f"No supported images found in {image_folder}")
        sys.exit(1)

    print(f"Found {len(image_files)} images. Processing...")

    annotations = {}
    for idx, img_path in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] Annotating: {img_path.name}")
        annotation = annotate_image(img_path)
        annotations[img_path.name] = {
            "path": str(img_path),
            "annotation": annotation,
        }

    # Save annotations to JSON
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"\nAnnotations saved to {output_file}")


if __name__ == "__main__":
    main()
