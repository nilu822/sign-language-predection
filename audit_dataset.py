import os
import cv2
import numpy as np

# === Settings ===
DATASET_DIR = "dataset"
MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 220
MIN_IMAGE_COUNT = 50
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')

print("\n📊 DATASET AUDIT STARTED\n")

def check_image_quality(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return "Unreadable", None
        if img.shape[0] < 50 or img.shape[1] < 50:
            return "Too Small", None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < MIN_BRIGHTNESS:
            return f"Too Dark ({int(brightness)})", brightness
        elif brightness > MAX_BRIGHTNESS:
            return f"Too Bright ({int(brightness)})", brightness
        return "OK", brightness
    except:
        return "Error", None

total_images = 0
bad_images = []
brightness_per_class = {}

# Check 'alphabet' and 'word' folders
for group in ['alphabet', 'word']:
    group_path = os.path.join(DATASET_DIR, group)
    if not os.path.exists(group_path):
        continue

    print(f"🔍 Scanning: {group}/")
    for class_name in sorted(os.listdir(group_path)):
        class_path = os.path.join(group_path, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(IMG_EXTENSIONS)]
        count = len(images)
        print(f"  • Class '{class_name}': {count} images", end="")

        if count < MIN_IMAGE_COUNT:
            print(f" ⚠️ [Low Sample Count! Expected ≥ {MIN_IMAGE_COUNT}]")
        else:
            print()

        class_brightness = []

        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            result, brightness = check_image_quality(img_path)
            total_images += 1
            if result != "OK":
                bad_images.append((img_path, result))
            if brightness is not None:
                class_brightness.append(brightness)

        # Store brightness stats
        if class_brightness:
            avg = np.mean(class_brightness)
            std = np.std(class_brightness)
            brightness_per_class[f"{group}/{class_name}"] = (avg, std)

# === Final Report ===
print("\n✅ Audit Summary:")
print(f"  Total images checked: {total_images}")
print(f"  Problematic images found: {len(bad_images)}")

if bad_images:
    print("\n⚠️ Problem Details:")
    for path, issue in bad_images[:20]:  # Limit to first 20 for readability
        print(f"  - {path} → {issue}")
    if len(bad_images) > 20:
        print(f"  ...and {len(bad_images) - 20} more.")

# === Brightness Report ===
if brightness_per_class:
    print("\n💡 Class-wise Brightness Statistics:")
    print(f"{'Class':<40} {'Avg Brightness':>16} {'Std Dev':>10}")
    print("-" * 70)
    for cls, (avg, std) in brightness_per_class.items():
        print(f"{cls:<40} {avg:>16.2f} {std:>10.2f}")

print("\n✅ DATASET AUDIT COMPLETE\n")
