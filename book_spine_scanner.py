"""
Book Spine Scanner
==================
An AI-powered computer vision system for detecting, segmenting, analyzing, 
and cataloging books on a shelf using YOLO object detection and OpenCV spine analysis.

Author: Antigravity AI
License: MIT
"""

import argparse
import json
import os
import sys
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO


class BookSpine:
    """Represents a single detected book spine with physical attributes."""

    def __init__(self, book_id, bbox, confidence, spine_color_hsv, image_shape):
        self.book_id = book_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = round(float(confidence), 3)
        
        x1, y1, x2, y2 = bbox
        self.width = x2 - x1
        self.height = y2 - y1
        self.center_x = (x1 + x2) // 2
        self.center_y = (y1 + y2) // 2
        self.aspect_ratio = round(self.height / max(1, self.width), 2)
        
        # Categorize Spine Thickness
        if self.width < 30:
            self.thickness_category = "Slim"
        elif self.width < 60:
            self.thickness_category = "Medium"
        else:
            self.thickness_category = "Thick (Volume)"

        # Color signature (BGR & Hex)
        self.bgr_color = [int(c) for c in spine_color_hsv["bgr"]]
        self.hex_color = f"#{self.bgr_color[2]:02x}{self.bgr_color[1]:02x}{self.bgr_color[0]:02x}"
        self.color_name = spine_color_hsv["color_name"]

    def to_dict(self):
        """Converts book spine data to dictionary for JSON export."""
        return {
            "book_id": self.book_id,
            "confidence": self.confidence,
            "position": {
                "x1": self.bbox[0],
                "y1": self.bbox[1],
                "x2": self.bbox[2],
                "y2": self.bbox[3],
                "center_x": self.center_x,
                "center_y": self.center_y
            },
            "spine_dimensions": {
                "width_px": self.width,
                "height_px": self.height,
                "aspect_ratio": self.aspect_ratio,
                "thickness_category": self.thickness_category
            },
            "spine_color": {
                "hex": self.hex_color,
                "bgr": self.bgr_color,
                "color_name": self.color_name
            }
        }


class BookSpineScanner:
    """Core scanner engine for bookshelf object detection and spine feature extraction."""

    def __init__(self, model_weights="yolo11n.pt", conf_threshold=0.35):
        self.conf_threshold = conf_threshold
        print(f"[INFO] Initializing Book Spine Scanner with YOLO model: {model_weights}...")
        self.model = YOLO(model_weights)

    def _extract_dominant_color(self, img, bbox):
        """Extracts dominant color profile from a book spine region."""
        x1, y1, x2, y2 = bbox
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return {"bgr": (128, 128, 128), "color_name": "Unknown"}

        # Calculate mean color in BGR
        mean_bgr = cv2.mean(crop)[:3]
        b, g, r = mean_bgr

        # Convert to HSV to estimate color name
        hsv_pixel = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h_val, s_val, v_val = hsv_pixel

        if v_val < 50:
            color_name = "Black / Dark"
        elif s_val < 30 and v_val > 200:
            color_name = "White / Light"
        elif s_val < 30:
            color_name = "Grey"
        elif h_val < 15 or h_val > 165:
            color_name = "Red / Maroon"
        elif 15 <= h_val < 35:
            color_name = "Orange / Brown"
        elif 35 <= h_val < 85:
            color_name = "Green"
        elif 85 <= h_val < 135:
            color_name = "Blue"
        else:
            color_name = "Purple"

        return {"bgr": mean_bgr, "color_name": color_name}

    def process_bookshelf(self, image_path, output_dir="output", show_display=False):
        """Scans a bookshelf image, detects books, catalogs spines, and saves reports."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Bookshelf image not found at: {image_path}")

        os.makedirs(output_dir, exist_ok=True)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image at: {image_path}")

        h, w, _ = img.shape
        print(f"[INFO] Scanning bookshelf image '{image_path}' ({w}x{h} px)...")

        # Run YOLO object detection
        results = self.model(img, conf=self.conf_threshold)[0]

        raw_books = []

        # Filter detections for books
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]

            # Focus on 'book' class or similar container detections
            if class_name in ["book", "cell phone"]:  # Treat rectangular objects as book spines
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = [x1, y1, x2, y2]
                raw_books.append((bbox, conf))

        # Sort detected books from Left to Right based on center_x
        raw_books.sort(key=lambda b: (b[0][0] + b[0][2]) // 2)

        # Create BookSpine objects
        scanned_books = []
        for idx, (bbox, conf) in enumerate(raw_books, start=1):
            color_info = self._extract_dominant_color(img, bbox)
            book = BookSpine(
                book_id=idx,
                bbox=bbox,
                confidence=conf,
                spine_color_hsv=color_info,
                image_shape=(h, w)
            )
            scanned_books.append(book)

        # Draw Visual Annotations on Output Image
        annotated_img = img.copy()

        # Draw Global HUD Header Banner
        cv2.rectangle(annotated_img, (0, 0), (w, 45), (40, 40, 40), -1)
        header_text = f"BOOK SPINE SCANNER | TOTAL BOOKS DETECTED: {len(scanned_books)}"
        cv2.putText(
            annotated_img, header_text, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2
        )

        for book in scanned_books:
            x1, y1, x2, y2 = book.bbox
            color = (0, 255, 120)  # Emerald green for detected spine

            # Draw Spine Bounding Box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated_img, (book.center_x, book.center_y), 4, (0, 0, 255), -1)

            # Draw Book Index Tag on Top of Bounding Box
            tag_text = f"Book #{book.book_id}"
            cv2.rectangle(annotated_img, (x1, max(0, y1 - 22)), (x1 + 75, y1), color, -1)
            cv2.putText(
                annotated_img, tag_text, (x1 + 5, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2
            )

            # Draw Spine Feature Label at Bottom of Bounding Box
            sub_label = f"{book.thickness_category} ({book.width}px)"
            cv2.putText(
                annotated_img, sub_label, (x1, min(h - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )

        # Save Output Annotated Image
        output_image_path = os.path.join(output_dir, "scanned_bookshelf.jpg")
        cv2.imwrite(output_image_path, annotated_img)
        print(f"[SUCCESS] Saved annotated scan to: {output_image_path}")

        # Construct JSON Catalog & Text Summary Report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        thickness_counts = {}
        for b in scanned_books:
            thickness_counts[b.thickness_category] = thickness_counts.get(b.thickness_category, 0) + 1

        catalog_data = {
            "timestamp": timestamp,
            "image_scanned": os.path.basename(image_path),
            "total_books_detected": len(scanned_books),
            "thickness_breakdown": thickness_counts,
            "books": [b.to_dict() for b in scanned_books]
        }

        # Save JSON Catalog Database
        json_path = os.path.join(output_dir, "book_catalog.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(catalog_data, f, indent=2)
        print(f"[SUCCESS] Saved book catalog database to: {json_path}")

        # Save Human-Readable Text Summary Report
        text_report_path = os.path.join(output_dir, "library_report.txt")
        with open(text_report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("BOOKSHELF INVENTORY & SPINE SCAN REPORT\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Image File: {os.path.basename(image_path)}\n")
            f.write(f"Total Books Scanned: {len(scanned_books)}\n")
            f.write("=" * 60 + "\n\n")

            f.write("SPINE THICKNESS BREAKDOWN:\n")
            for cat, count in thickness_counts.items():
                f.write(f"  - {cat}: {count} books\n")
            f.write("-" * 40 + "\n\n")

            f.write("DETECTED BOOK CATALOG (LEFT TO RIGHT ORDER):\n")
            for b in scanned_books:
                f.write(
                    f"  Book #{b.book_id:02d}: {b.thickness_category} Spine "
                    f"({b.width}x{b.height} px) | Color: {b.color_name} ({b.hex_color}) "
                    f"| Pos: ({b.bbox[0]}, {b.bbox[1]})\n"
                )
            f.write("-" * 60 + "\n")

        print(f"[SUCCESS] Saved text report to: {text_report_path}")

        # Console Summary
        self._print_console_summary(catalog_data)

        # GUI Display
        if show_display:
            cv2.imshow("Book Spine Scanner", annotated_img)
            print("[INFO] Press any key on the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return catalog_data

    def _print_console_summary(self, catalog):
        """Prints formatted scan summary to console."""
        print("\n" + "=" * 60)
        print("               BOOK SPINE SCANNER CATALOG")
        print("=" * 60)
        print(f" Timestamp:           {catalog['timestamp']}")
        print(f" Total Books Scanned: {catalog['total_books_detected']}")
        print(f" Spine Thickness:     {catalog['thickness_breakdown']}")
        print("-" * 60)

        for b in catalog["books"]:
            print(
                f" [Book #{b['book_id']:02d}] Spine: {b['spine_dimensions']['thickness_category']} "
                f"({b['spine_dimensions']['width_px']}x{b['spine_dimensions']['height_px']} px) | "
                f"Color: {b['spine_color']['color_name']} | Conf: {b['confidence']:.2f}"
            )
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Book Spine Scanner using YOLO & OpenCV")
    parser.add_argument("--image", type=str, default="input/bookshelf.jpg", help="Path to input bookshelf image")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="YOLO model weights file")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")

    args = parser.parse_args()

    scanner = BookSpineScanner(model_weights=args.weights, conf_threshold=args.conf)
    scanner.process_bookshelf(
        image_path=args.image,
        output_dir=args.output_dir,
        show_display=not args.headless
    )


if __name__ == "__main__":
    main()
