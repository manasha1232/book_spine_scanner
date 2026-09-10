# Book Spine Scanner 📚🔍

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

An AI-powered computer vision project for **automated library inventory, bookshelf digitization, and book spine scanning**. Powered by **YOLOv11** and **OpenCV**, this tool detects standing books on shelves, measures physical spine dimensions (width/thickness, height, aspect ratio), extracts dominant color signatures, orders books sequentially from left to right, and exports structured library databases.

---

## 🌟 Key Features

- 📖 **YOLOv11 Book Detection**: Locates individual books and volumes across bookshelves with high precision.
- 📐 **Physical Spine Analysis**: Calculates spine width (px), height (px), aspect ratios, and classifies volumes into thickness categories (*Slim*, *Medium*, *Thick Volume*).
- 🎨 **Color Signature Extraction**: Analyzes BGR & HSV color spaces to determine the dominant spine color signature and HEX code.
- 🗂️ **Sequential Shelf Ordering**: Sorts scanned books in natural physical sequence (Left-to-Right / Top-to-Bottom).
- 🎨 **Rich HUD Visual Annotations**:
  - Green bounding boxes with indexed book ID tags (`Book #1`, `Book #2`, ...).
  - Thickness labels & center-point targets overlaid on the scanned image.
  - Global status banner showing total count.
- 📊 **Multi-Format Reporting**:
  - `output/scanned_bookshelf.jpg`: Visual image with spine annotations.
  - `output/book_catalog.json`: Machine-readable JSON database of all detected volumes.
  - `output/library_report.txt`: Human-readable shelf inventory report.

---

## 📂 Project Structure

```text
book_spine_scanner/
├── book_spine_scanner.py       # Main scanner engine & detection pipeline
├── generate_bookshelf_demo.py  # Test bookshelf image generator
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── input/
│   └── bookshelf.jpg           # Input bookshelf image
└── output/
    ├── scanned_bookshelf.jpg   # Visual output with spine overlays
    ├── book_catalog.json       # Exported JSON library database
    └── library_report.txt      # Text summary report
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.9+ is installed on your system.

### 2. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/book-spine-scanner.git
cd book-spine-scanner

pip install -r requirements.txt
```

---

## ⚡ Quickstart

### Step 1: Fetch/Generate Sample Bookshelf Image
Run the demo script to obtain a sample bookshelf test image (`input/bookshelf.jpg`):

```bash
python generate_bookshelf_demo.py
```

### Step 2: Run the Book Spine Scanner

**Run in Headless Mode (saves output files to `output/`):**
```bash
python book_spine_scanner.py --image input/bookshelf.jpg --headless
```

**Run in Interactive GUI Mode (opens OpenCV image window):**
```bash
python book_spine_scanner.py --image input/bookshelf.jpg
```

---

## 📋 Sample Output

### 1. Terminal Console Summary
```text
============================================================
               BOOK SPINE SCANNER CATALOG
============================================================
 Timestamp:           2026-09-10 20:39:50
 Total Books Scanned: 13
 Spine Thickness:     {'Slim': 4, 'Medium': 7, 'Thick (Volume)': 2}
------------------------------------------------------------
 [Book #01] Spine: Slim (28x185 px) | Color: Blue | Conf: 0.76
 [Book #02] Spine: Medium (42x190 px) | Color: Red / Maroon | Conf: 0.72
 [Book #03] Spine: Thick (Volume) (65x195 px) | Color: Green | Conf: 0.67
 ...
============================================================
```

### 2. JSON Catalog Database (`output/book_catalog.json`)
```json
{
  "timestamp": "2026-09-10 20:39:50",
  "image_scanned": "bookshelf.jpg",
  "total_books_detected": 13,
  "thickness_breakdown": {
    "Slim": 4,
    "Medium": 7,
    "Thick (Volume)": 2
  },
  "books": [
    {
      "book_id": 1,
      "confidence": 0.76,
      "position": {
        "x1": 52,
        "y1": 110,
        "x2": 80,
        "y2": 295,
        "center_x": 66,
        "center_y": 202
      },
      "spine_dimensions": {
        "width_px": 28,
        "height_px": 185,
        "aspect_ratio": 6.61,
        "thickness_category": "Slim"
      },
      "spine_color": {
        "hex": "#284a9e",
        "bgr": [158, 74, 40],
        "color_name": "Blue"
      }
    }
  ]
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
