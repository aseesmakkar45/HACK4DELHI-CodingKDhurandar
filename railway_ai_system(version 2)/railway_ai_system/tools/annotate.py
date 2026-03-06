"""
RailDrishti AI — Knowledge Distillation Annotation Oracle
==========================================================
Uses Gemini Vision API to automatically annotate images
as YOLO-format label files.

Each image gets a corresponding .txt file containing:
  <class_id> <cx> <cy> <width> <height>   (normalized 0-1)

Classes:
  0: tamper_action
  1: person_passing
  2: maintenance_worker
  3: animal
  4: tool_on_track

Usage:
  1. Put raw images in: dataset/images/raw/
  2. Run: python tools/annotate.py
  3. Annotated label .txt files appear in: dataset/labels/raw/
  4. Move images + labels into train/val splits
  5. Run: python tools/train.py
"""

import os
import json
import base64
import argparse
from pathlib import Path

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

CLASSES = {
    "tamper_action":      0,
    "person_passing":     1,
    "maintenance_worker": 2,
    "animal":             3,
    "tool_on_track":      4,
}

ANNOTATION_PROMPT = """
You are an expert vision analyst for Indian Railways security.
Analyze this image and identify all relevant objects.

For each object, output a JSON array where each element is:
{
  "class": "<one of: tamper_action, person_passing, maintenance_worker, animal, tool_on_track>",
  "confidence": <float 0-1>,
  "bbox_normalized": [cx, cy, width, height]   // all values 0-1 relative to image
}

Classification rules:
- tamper_action: A person crouching/bending, using a tool (bolt cutter, crowbar, grinder, hammer) 
  on track hardware (rails, fishplates, bolts, clips, signal cables). This is the most important class.
- person_passing: A person standing upright, walking across or near the track normally.
- maintenance_worker: A person in PPE (vest, helmet, safety gear) performing track maintenance.
- animal: Any animal (cow, dog, goat, bird, etc.) on or near the track.
- tool_on_track: A tool visibly placed on or in contact with track hardware (even without a person).

If the image shows no relevant objects, return an empty array: []

Return ONLY valid JSON, no explanation.
"""


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_gemini_vision(image_path: str) -> list:
    """Call Gemini Vision API to get annotations."""
    if not GEMINI_API_KEY:
        print(f"  [SKIP] No GEMINI_API_KEY set. Generating placeholder label.")
        return []

    try:
        import urllib.request, urllib.error
        import json as _json

        b64 = encode_image(image_path)
        ext = Path(image_path).suffix.lower().replace(".", "")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")

        payload = {
            "contents": [{
                "parts": [
                    {"text": ANNOTATION_PROMPT},
                    {"inline_data": {"mime_type": mime, "data": b64}}
                ]
            }]
        }

        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}")

        req = urllib.request.Request(
            url,
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read())

        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        annotations = _json.loads(text)
        return annotations

    except Exception as e:
        print(f"  [ERROR] Gemini API: {e}")
        return []


def write_yolo_label(annotations: list, label_path: str):
    """Write YOLO-format label file."""
    lines = []
    for ann in annotations:
        cls_name = ann.get("class", "")
        cls_id   = CLASSES.get(cls_name, -1)
        if cls_id < 0:
            continue
        conf = ann.get("confidence", 0)
        if conf < 0.4:
            continue
        bbox = ann.get("bbox_normalized", [])
        if len(bbox) != 4:
            continue
        cx, cy, w, h = [max(0, min(1, float(v))) for v in bbox]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(lines))

    return len(lines)


def annotate_folder(input_dir: str, output_dir: str):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = list(input_path.glob("*.jpg")) + \
             list(input_path.glob("*.jpeg")) + \
             list(input_path.glob("*.png"))

    print(f"\nRailDrishti AI — Annotation Oracle")
    print(f"Images found : {len(images)}")
    print(f"Output dir   : {output_path}\n")

    success, skipped = 0, 0

    for i, img_path in enumerate(images):
        label_path = output_path / (img_path.stem + ".txt")
        print(f"[{i+1}/{len(images)}] {img_path.name} ... ", end="", flush=True)

        annotations = call_gemini_vision(str(img_path))
        n = write_yolo_label(annotations, str(label_path))

        if n > 0:
            print(f"{n} objects labeled ✓")
            success += 1
        else:
            print(f"no objects detected")
            skipped += 1

    print(f"\n{'='*50}")
    print(f"Annotation complete: {success} labeled, {skipped} empty")
    print(f"Next step: python tools/train.py")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RailDrishti AI — Annotation Oracle")
    parser.add_argument("--input",  default="dataset/images/raw",   help="Input images dir")
    parser.add_argument("--output", default="dataset/labels/raw",   help="Output labels dir")
    args = parser.parse_args()
    annotate_folder(args.input, args.output)
