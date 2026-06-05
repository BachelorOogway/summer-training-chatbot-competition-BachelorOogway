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
        
        example_annotation1 = """A group portrait of a Hong Kong RC aircraft modeling club outdoors on a sunny clear day. Around 25 hobbyists (mixed male and female adults, most wearing matching dark navy baseball caps) gather around a large custom-built twin-wing gasoline-powered radio-controlled aerobatic model plane as the central focus.
The oversized RC biplane features a red fuselage, white nose cone, and wings painted in white with bold yellow, blue and red decorative stripes; partial "BOEING" lettering is printed on both upper wing surfaces. Multiple participants hold up the plane's wings with their hands, and nearly all people give a thumbs-up gesture to mark a successful test flight or club event.
The group lines up in two rows: front row kneels on paved asphalt aircraft taxiway, back row stands on the pavement bordered by green lawn. In the background: dense high-rise residential apartment towers of Tin Shui Wai (New Territories, Hong Kong) rise on the left, with thick green woodland hillside stretching across the right under a cloudless bright blue sky. The scene captures a celebratory post-flight team photo for radio-controlled aviation enthusiasts."""

        example_annotation2 = """A close-up indoor shot capturing a fluffy golden British Shorthair (golden chinchilla cat) having its round cheeks gently squished between two human thumbs and fingers.
The cat features rich warm orange-gold top fur transitioning to creamy off-white fur on its muzzle, chest and neck, large glossy round dark emerald-green eyes, a tiny pale pink nose, and fine white whiskers extending outward. Its plush, thick double coat appears extremely soft and voluminous.
Two light-skinned human hands frame the cat's face from left and right sides, pressing its cheek fur outwards into a rounded, squishy shape. The background consists of muted grey woven textured carpet flooring and a plain pale white baseboard along the upper edge of the frame. A small faint white watermark reading "MAPLE.CAT" sits on the bottom-left carpet area. The cat has a calm, wide-eyed neutral facial expression under gentle cheek pinching."""

        example_annotation3 = """A formal indoor event photograph taken at a ceremony dated December 2021, featuring two masked adult Asian professionals engaged in face-to-face conversation against a digital starry-blue backdrop screen with partial golden "December 2021" text visible at the top left.
The male figure on the left wears a tailored dark navy formal suit, white collared dress shirt, patterned necktie, and a white disposable face mask covering his nose and mouth; a small boutonniere flower pin adorns his left lapel. He gestures expressively with both open palms forward while speaking toward the woman beside him.
The female figure on the right sports textured off-white cream tweed blazer over a black turtleneck top, thin-rimmed eyeglasses, matching white protective face mask, a pinned name badge, and a decorative peach rose boutonniere tied with lavender ribbon on her blazer lapel. She stands attentively facing the man, listening intently.
The background LED wall has a deep navy base dotted with scattered pale white speckles resembling starry night sky, establishing a formal corporate or institutional award/inauguration ceremony atmosphere."""
        
        response = gpt_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Please provide a detailed annotation/description of this image, similar in style and detail level to these examples:

EXAMPLE 1 (Group/Event Scene):
{example_annotation1}

EXAMPLE 2 (Close-up Subject):
{example_annotation2}

EXAMPLE 3 (Formal Event/Professional Interaction):
{example_annotation3}

For the image provided, include: what is shown, key objects, colors, layout, any text visible, people/groups/animals, location context, and overall scene composition. Be thorough and descriptive.""",
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
            max_tokens=800,
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
