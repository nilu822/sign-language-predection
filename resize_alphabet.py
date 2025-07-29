import os
import cv2

def resize_and_rename_images():
    base_path = 'dataset/alphabet'
    target_size = (96, 96)
    max_images = 100

    print(f"[INFO] Processing folders inside: {base_path}")

    for class_name in sorted(os.listdir(base_path)):
        class_path = os.path.join(base_path, class_name)

        if not os.path.isdir(class_path):
            continue

        print(f"[INFO] Processing: {class_name}")

        # Get all image files and sort them
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()

        # Only keep the first 100
        image_files = image_files[:max_images]

        # Resize and rename
        for idx, filename in enumerate(image_files):
            full_path = os.path.join(class_path, filename)
            img = cv2.imread(full_path)

            if img is None:
                print(f"[WARNING] Couldn't read: {filename}")
                continue

            resized = cv2.resize(img, target_size)

            # New filename: A_0.jpg, B_99.jpg, etc.
            new_filename = f"{class_name}_{idx}.jpg"
            new_path = os.path.join(class_path, new_filename)

            # Save resized image
            cv2.imwrite(new_path, resized)

            # Remove original if name has changed
            if filename != new_filename:
                try:
                    os.remove(full_path)
                except Exception as e:
                    print(f"[ERROR] Couldn't delete old file {filename}: {e}")

        # Delete leftover images (beyond first 100)
        remaining_files = os.listdir(class_path)
        for file in remaining_files:
            if not file.startswith(f"{class_name}_") or int(file.split('_')[-1].split('.')[0]) >= max_images:
                try:
                    os.remove(os.path.join(class_path, file))
                except:
                    pass

        print(f"[DONE] {class_name}: Resized and saved 100 images as {class_name}_0.jpg to {class_name}_99.jpg")

    print("[✅ COMPLETE] All alphabet folders resized and renamed.")

if __name__ == "__main__":
    resize_and_rename_images()














# resizing to 96x96 having skin masking ----
# import os
# import cv2
# import numpy as np

# def apply_skin_mask(img):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#     upper_skin = np.array([20, 255, 255], dtype=np.uint8)
#     mask = cv2.inRange(hsv, lower_skin, upper_skin)
#     return mask  # Binary output: white hand, black background

# def convert_and_mask_alphabet_images(dataset_path='dataset/alphabet', target_size=(96, 96)):
#     print("[INFO] Converting alphabet images to 96x96 and applying skin mask...")
    
#     for letter in sorted(os.listdir(dataset_path)):
#         folder = os.path.join(dataset_path, letter)
#         if not os.path.isdir(folder):
#             continue
#         print(f"[PROCESSING] Folder: {folder}")

#         for filename in os.listdir(folder):
#             img_path = os.path.join(folder, filename)
#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f"[WARNING] Could not read {img_path}")
#                 continue

#             resized = cv2.resize(img, target_size)
#             mask = apply_skin_mask(resized)
#             cv2.imwrite(img_path, mask)

#     print("[DONE] All images converted and updated.")

# if __name__ == "__main__":
#     convert_and_mask_alphabet_images()






















# Resize all alphabet gesture images to 96x96 (ASL)
# Only keep and resize up to 100 images per alphabet class

# Resize all alphabet gesture images to 96x96 (ASL)
# Only keep and resize up to 100 images per class (delete the rest)

# import os
# import cv2

# def resize_and_rename_alphabet_images():
#     folder_path = 'dataset/alphabet'
#     print("[INFO] Processing:", folder_path)

#     for class_name in os.listdir(folder_path):
#         class_path = os.path.join(folder_path, class_name)
#         if not os.path.isdir(class_path):
#             continue

#         image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]
#         image_files.sort()  # Sort files for consistent selection

#         # Limit to first 100
#         image_files = image_files[:100]

#         # Process images: resize & rename
#         for idx, file in enumerate(image_files):
#             original_path = os.path.join(class_path, file)
#             img = cv2.imread(original_path)
#             if img is None:
#                 continue

#             resized = cv2.resize(img, (96, 96))
#             new_filename = f"{class_name}_{idx}.jpg"
#             new_path = os.path.join(class_path, new_filename)

#             cv2.imwrite(new_path, resized)

#             # If filename is different, delete the original
#             if file != new_filename:
#                 os.remove(original_path)

#         # Delete any leftover files beyond 100
#         all_files_after = os.listdir(class_path)
#         for file in all_files_after:
#             if not file.startswith(f"{class_name}_") or int(file.split('_')[-1].split('.')[0]) >= 100:
#                 try:
#                     os.remove(os.path.join(class_path, file))
#                 except Exception:
#                     pass

#         print(f"[DONE] {class_name}: saved 100 resized images as {class_name}_0 to {class_name}_99")

#     print("[✅ COMPLETE] All alphabet folders processed.")

# if __name__ == "__main__":
#     resize_and_rename_alphabet_images()
















# # Here it is resizing the images ASL ( American Sign Language )
# #Converting it to 96X96 dimensions


# import cv2
# import os

# def resize_alphabet_images():
#     folder_path = 'dataset/alphabet'
#     print("[INFO] Resizing images in:", folder_path)
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.endswith(('.jpg', '.png')):
#                 full_path = os.path.join(root, file)
#                 img = cv2.imread(full_path)
#                 if img is not None:
#                     resized = cv2.resize(img, (96, 96))
#                     cv2.imwrite(full_path, resized)
#     print("[DONE] All alphabet images resized to 96x96.")

# resize_alphabet_images()
