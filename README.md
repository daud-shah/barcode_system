# Barcode Orientation Correction & Decoding System

Automatically detects, corrects orientation, crops, and decodes barcodes
from images. Works on single images or full folders.

---

## Setup

### 1. Create virtual environment
```bash
python -m venv barcode_env
```

### 2. Activate virtual environment
```bash
# Windows
barcode_env\Scripts\activate

# Linux / Mac
source barcode_env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Single image
```bash
python run.py --input D:\images\photo.jpeg
```

### Full folder
```bash
python run.py --input D:\images\
```

### Interactive (no arguments)
```bash
python run.py
# Then type the path when prompted
```

---

## Output Structure

```
output/
├── crops/
│   ├── photo_crop_1.jpeg     ← cropped + upright barcode
│   ├── photo_crop_2.jpeg
│   └── ...
├── annotated/
│   ├── photo_annotated.jpeg  ← original image with boxes + decoded values
│   └── ...
└── decoded_barcodes.json     ← all results in JSON format
```

---

## JSON Output Format

```json
{
    "timestamp": "2026-06-29 14:30:00",
    "input": "D:/images/",
    "total_images": 17,
    "total_crops": 79,
    "decoded": 74,
    "failed": 5,
    "unique_values": ["01001011", "130-1244", "ABC-1344", ...],
    "results": [
        {
            "source_image": "photo.jpeg",
            "crop_file":    "photo_crop_1.jpeg",
            "barcode_id":   1,
            "angle":        -17.7,
            "decoded":      "01001011",
            "status":       "success"
        },
        ...
    ]
}
```

---

## How It Works

1. **Detection** — tries 7 preprocessings × 7 scales = 49 attempts per image
2. **Annotation** — draws green boxes with decoded values on original image
3. **Crop** — uses `minAreaRect` to find exact angle, rotates full image, crops with extra bottom padding for number text
4. **Decode** — tries 4 rotations × 5 preprocessings × 4 scales using `zxing-cpp`
5. **JSON** — saves all results with metadata

---

## Requirements

- Python 3.10+
- No additional DLL or system library needed on Windows
