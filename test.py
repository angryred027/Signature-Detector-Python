import cv2
import numpy as np
import os

input_dir = "./inputs"
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    orig = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contour found for {filename}")
        continue

    largest_contour = max(contours, key=cv2.contourArea)

    green_edges = np.zeros_like(img)
    cv2.drawContours(green_edges, [largest_contour], -1, (0, 255, 0), 10)

    overlay = cv2.addWeighted(orig, 0.8, green_edges, 0.8, 0)

    h, w = overlay.shape[:2]
    new_h = 800
    new_w = int(w * new_h / h)
    overlay_resized = cv2.resize(overlay, (new_w, new_h))

    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, overlay)
    print(f"Processed {filename} → {output_path}")
