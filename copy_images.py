import os
import shutil

def copy_images(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for img_file in os.listdir(src_dir):
        if img_file.endswith('.jpg'):
            src_file = os.path.join(src_dir, img_file)
            dest_file = os.path.join(dest_dir, img_file)
            shutil.copyfile(src_file, dest_file)
    print(f"Copied images from {src_dir} to {dest_dir}")

# Base path for the dataset
base_path = 'D:/EmotionDetector/dataset'

# Paths to the original YOLOv8 dataset directories
yolo_train_images = os.path.join(base_path, 'original dataset yolov8 format/train/images')
yolo_valid_images = os.path.join(base_path, 'original dataset yolov8 format/valid/images')
yolo_test_images = os.path.join(base_path, 'original dataset yolov8 format/test/images')

# Paths to the CSV TensorFlow dataset directories
csv_train_images = os.path.join(base_path, 'csv tensorflow with augmentations/train/images')
csv_valid_images = os.path.join(base_path, 'csv tensorflow with augmentations/valid/images')
csv_test_images = os.path.join(base_path, 'csv tensorflow with augmentations/test/images')

# Copy images to the respective directories
copy_images(yolo_train_images, csv_train_images)
copy_images(yolo_valid_images, csv_valid_images)
copy_images(yolo_test_images, csv_test_images)
