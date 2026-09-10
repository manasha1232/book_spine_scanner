"""
Bookshelf Demo Generator
========================
Generates a realistic test bookshelf image (input/bookshelf.jpg)
for out-of-the-box scanning with the Book Spine Scanner.
"""

import os
import urllib.request
import cv2
import numpy as np


def generate_bookshelf_demo(output_path="input/bookshelf.jpg"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Photographic bookshelf asset from Unsplash
    url = "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=800"
    
    print("[DEMO] Fetching photographic bookshelf test image...")
    try:
        urllib.request.urlretrieve(url, output_path)
        img = cv2.imread(output_path)
        if img is not None:
            print(f"[DEMO] Successfully fetched bookshelf photo ({img.shape[1]}x{img.shape[0]} px) at: {output_path}")
            return
    except Exception as e:
        print(f"[WARN] Failed to fetch image online: {e}. Falling back to synthetic generation...")

    # Synthetic fallback: 600x400 bookshelf canvas
    canvas = np.ones((400, 600, 3), dtype=np.uint8) * 230
    
    # Draw Wooden Shelf Plank
    cv2.rectangle(canvas, (0, 350), (600, 370), (60, 100, 150), -1)
    cv2.rectangle(canvas, (0, 350), (600, 370), (30, 50, 80), 2)

    # Draw standing book spines with various widths and colors
    books = [
        (30, 240, 50, (40, 40, 180)),    # Red Book
        (85, 260, 45, (40, 160, 40)),   # Green Book
        (135, 230, 60, (180, 80, 40)),   # Blue Book
        (200, 270, 35, (40, 180, 200)),  # Yellow Book
        (240, 250, 55, (160, 40, 160)),  # Purple Book
        (300, 265, 40, (80, 80, 80)),    # Dark Grey Book
        (345, 235, 65, (50, 120, 220)),  # Orange Book
        (415, 255, 30, (200, 100, 50)),  # Cyan Book
    ]

    for x1, height, width, color in books:
        y1 = 350 - height
        x2 = x1 + width
        cv2.rectangle(canvas, (x1, y1), (x2, 350), color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, 350), (20, 20, 20), 2)
        # Spine line decoration
        cv2.line(canvas, (x1 + 6, y1 + 10), (x1 + 6, 340), (240, 240, 240), 1)

    cv2.imwrite(output_path, canvas)
    print(f"[DEMO] Saved synthetic bookshelf image to: {output_path}")


if __name__ == "__main__":
    generate_bookshelf_demo()
