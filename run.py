"""
Barcode Orientation Correction & Decoding System
=================================================
Usage:
    python run.py --input <image_path or folder_path>

Examples:
    python run.py --input photo.jpg
    python run.py --input D:/images/

Output structure:
    output/
    ├── crops/          -> cropped + orientation-corrected barcodes
    ├── annotated/      -> original images with barcode boxes drawn
    └── decoded_barcodes.json
"""

import cv2
import numpy as np
import os
import json
import argparse
import zxingcpp
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────
#  SUPPORTED IMAGE EXTENSIONS
# ─────────────────────────────────────────────
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─────────────────────────────────────────────
#  STEP 1: DETECT ALL BARCODES IN AN IMAGE
# ─────────────────────────────────────────────

def detect_all_barcodes(image: np.ndarray) -> list:
    """
    Detect barcode corner locations using multiple scales and
    preprocessings to maximize detection rate.

    Returns list of corner arrays (each shape: Nx2).
    """
    detector    = cv2.barcode.BarcodeDetector()
    all_corners = []

    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel   = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp    = cv2.filter2D(gray, -1, kernel)
    _, otsu  = cv2.threshold(gray,     0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph    = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, morph_k)

    candidates = [
        image,
        cv2.cvtColor(gray,     cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(otsu,     cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(otsu2,    cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(sharp,    cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(morph,    cv2.COLOR_GRAY2BGR),
    ]

    scales = [1.5, 1.25, 1.0, 0.75, 0.5, 0.35, 2.0]

    for proc in candidates:
        for scale in scales:
            h, w   = proc.shape[:2]
            scaled = cv2.resize(proc,
                                (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_CUBIC)

            detected, corners = detector.detect(scaled)
            if not detected or corners is None:
                continue

            for corner in corners:
                orig_corner = corner / scale
                cx = np.mean(orig_corner[:, 0])
                cy = np.mean(orig_corner[:, 1])

                # Skip duplicates within 50px
                is_duplicate = False
                for existing in all_corners:
                    ex = np.mean(existing[:, 0])
                    ey = np.mean(existing[:, 1])
                    if np.sqrt((cx - ex)**2 + (cy - ey)**2) < 50:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    all_corners.append(orig_corner)

    return all_corners


# ─────────────────────────────────────────────
#  STEP 2: ANNOTATE ORIGINAL IMAGE
# ─────────────────────────────────────────────

def annotate_image(image: np.ndarray,
                   corners_list: list,
                   decoded_values: list) -> np.ndarray:
    """
    Draw bounding boxes and decoded values on the original image.

    Args:
        image:          Original BGR image
        corners_list:   List of corner arrays from detect_all_barcodes()
        decoded_values: Decoded string for each barcode (same order)

    Returns:
        Annotated image
    """
    annotated = image.copy()

    for corners, value in zip(corners_list, decoded_values):
        pts = corners.reshape(-1, 2).astype(np.int32)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(pts)

        # Draw green box around barcode
        cv2.rectangle(annotated,
                      (x - 10, y - 10),
                      (x + w + 10, y + h + 10),
                      (0, 200, 0), 3)

        # Draw filled label background
        label       = value if value else "unread"
        font        = cv2.FONT_HERSHEY_SIMPLEX
        font_scale  = 1.2
        thickness   = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        label_x = x - 10
        label_y = max(y - 20, th + 10)

        cv2.rectangle(annotated,
                      (label_x, label_y - th - 8),
                      (label_x + tw + 8, label_y + 4),
                      (0, 200, 0), -1)

        cv2.putText(annotated, label,
                    (label_x + 4, label_y),
                    font, font_scale,
                    (0, 0, 0), thickness,
                    cv2.LINE_AA)

    return annotated


# ─────────────────────────────────────────────
#  STEP 3: CROP EACH BARCODE
# ─────────────────────────────────────────────

def get_barcode_angle(corners: np.ndarray):
    pts  = corners.reshape(-1, 2).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        rw, rh = rh, rw
        angle  = angle + 90
    return angle, (cx, cy), (rw, rh)


def smart_crop(image: np.ndarray,
               corners: np.ndarray,
               extra: float = 0.6,
               padding: int = 60) -> tuple:
    """
    Rotate the full image to make the barcode upright,
    then crop with extra bottom space to include the number text.
    """
    angle, (cx, cy), (rw, rh) = get_barcode_angle(corners)
    img_h, img_w = image.shape[:2]

    M       = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (img_w, img_h),
                              flags=cv2.INTER_CUBIC,
                              borderValue=(255, 255, 255))

    x1 = int(cx - rw / 2 - padding)
    y1 = int(cy - rh / 2 - padding)
    x2 = int(cx + rw / 2 + padding)
    y2 = int(cy + rh / 2 + rh * extra + padding)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    crop = rotated[y1:y2, x1:x2]

    h, w = crop.shape[:2]
    if h > w * 1.2:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    return crop, angle


# ─────────────────────────────────────────────
#  STEP 4: DECODE BARCODE CROP
# ─────────────────────────────────────────────

def decode_crop(image: np.ndarray) -> str | None:
    """
    Try decoding the barcode crop at multiple rotations,
    preprocessings, and scales using zxing-cpp.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rotated = image
        elif angle == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        else:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        g        = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        enh      = clahe.apply(g)
        _, ot    = cv2.threshold(g,   0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, ot2   = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        for candidate in [rotated, g, enh, ot, ot2]:
            for scale in [1.0, 1.5, 2.0, 3.0]:
                h = candidate.shape[0]
                w = candidate.shape[1]
                sized = cv2.resize(candidate,
                                   (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_CUBIC)
                results = zxingcpp.read_barcodes(sized)
                for r in results:
                    val = r.text.strip()
                    if val:
                        return val
    return None


# ─────────────────────────────────────────────
#  MAIN PIPELINE: PROCESS ONE IMAGE
# ─────────────────────────────────────────────

def process_image(image_path: str,
                  crops_dir: str,
                  annotated_dir: str) -> list:
    """
    Full pipeline for a single image:
        Load → Detect → Annotate → Crop → Decode

    Returns list of result dicts for JSON.
    """
    # Load image
    image = cv2.imdecode(
        np.fromfile(image_path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    if image is None:
        print(f"  [ERROR] Cannot load: {image_path}")
        return []

    filename = Path(image_path).stem
    print(f"\n{'='*60}")
    print(f"  Image: {Path(image_path).name}")

    # Step 1: Detect all barcodes
    all_corners = detect_all_barcodes(image)
    print(f"  Detected: {len(all_corners)} barcodes")

    if not all_corners:
        # Save original as annotated with no detections
        ann_path = os.path.join(annotated_dir, f"{filename}_annotated.jpeg")
        cv2.imencode(".jpeg", image)[1].tofile(ann_path)
        return []

    # Step 2: Crop + decode each barcode
    results      = []
    decoded_vals = []

    for i, corners in enumerate(all_corners):
        crop_name = f"{filename}_crop_{i+1}.jpeg"
        crop_path = os.path.join(crops_dir, crop_name)

        try:
            crop, angle = smart_crop(image, corners)
            h, w = crop.shape[:2]

            if w < 80 or h < 80:
                decoded_vals.append(None)
                continue

            # Save crop
            cv2.imencode(".jpeg", crop)[1].tofile(crop_path)

            # Decode
            decoded = decode_crop(crop)
            decoded_vals.append(decoded)

            status = "✓" if decoded else "✗"
            print(f"    [{status}] crop_{i+1}  angle={angle:.1f}°  "
                  f"→  {decoded or 'Could not read'}")

            results.append({
                "source_image": Path(image_path).name,
                "crop_file":    crop_name,
                "barcode_id":   i + 1,
                "angle":        round(angle, 2),
                "decoded":      decoded if decoded else "Could not read",
                "status":       "success" if decoded else "failed"
            })

        except Exception as e:
            decoded_vals.append(None)
            print(f"    [!] crop_{i+1} error: {e}")

    # Step 3: Save annotated image
    annotated = annotate_image(image, all_corners, decoded_vals)
    ann_name  = f"{filename}_annotated.jpeg"
    ann_path  = os.path.join(annotated_dir, ann_name)
    cv2.imencode(".jpeg", annotated)[1].tofile(ann_path)
    print(f"  Annotated saved: {ann_name}")

    return results


# ─────────────────────────────────────────────
#  RUN PIPELINE ON INPUT (FILE OR FOLDER)
# ─────────────────────────────────────────────

def run(input_path: str) -> None:
    input_path = Path(input_path)

    # Collect image files
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED:
            print(f"Unsupported file type: {input_path.suffix}")
            return
        image_files = [str(input_path)]
        output_base = input_path.parent / "output"

    elif input_path.is_dir():
        image_files = [
            str(p) for p in sorted(input_path.iterdir())
            if p.suffix.lower() in SUPPORTED
        ]
        output_base = input_path / "output"

    else:
        print(f"Path does not exist: {input_path}")
        return

    if not image_files:
        print("No supported images found.")
        return

    # Create output folders
    crops_dir     = output_base / "crops"
    annotated_dir = output_base / "annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInput:      {input_path}")
    print(f"Images:     {len(image_files)}")
    print(f"Output:     {output_base}")

    # Process all images
    all_results = []
    for img_path in image_files:
        results = process_image(
            img_path,
            str(crops_dir),
            str(annotated_dir)
        )
        all_results.extend(results)

    # Save JSON
    success = sum(1 for r in all_results if r["status"] == "success")
    failed  = sum(1 for r in all_results if r["status"] == "failed")

    output_json = {
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input":         str(input_path),
        "total_images":  len(image_files),
        "total_crops":   len(all_results),
        "decoded":       success,
        "failed":        failed,
        "unique_values": sorted(set(
                            r["decoded"] for r in all_results
                            if r["status"] == "success"
                         )),
        "results":       all_results
    }

    json_path = output_base / "decoded_barcodes.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  Total crops:  {len(all_results)}")
    print(f"  Decoded:      {success}")
    print(f"  Failed:       {failed}")
    print(f"  Success rate: {success/max(len(all_results),1)*100:.1f}%")
    print(f"\n  Output folder:  {output_base}")
    print(f"  Crops:          {crops_dir}")
    print(f"  Annotated:      {annotated_dir}")
    print(f"  JSON:           {json_path}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Barcode Orientation Correction & Decoding System"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to a single image or a folder of images"
    )
    args = parser.parse_args()

    # If no argument given, ask in terminal
    if not args.input:
        print("\nBarcode Decoding System")
        print("-" * 40)
        path = input("Enter image path or folder path: ").strip().strip('"')
    else:
        path = args.input

    run(path)
