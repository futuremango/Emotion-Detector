import os
import pandas as pd

def generate_labels_csv(yolo_images_dir, yolo_labels_dir, output_csv_path):
    data = []
    for img_file in os.listdir(yolo_images_dir):
        if img_file.endswith('.jpg'):
            label_file = img_file.replace('.jpg', '.txt')
            label_path = os.path.join(yolo_labels_dir, label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) > 0:
                            class_id = int(parts[0])
                            data.append({'filename': img_file, 'label': class_id})

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv_path, index=False)
    else:
        print(f"No data found in {yolo_images_dir}, skipping {output_csv_path}")

# Base path for the dataset
base_path = 'D:/EmotionDetector/dataset'

# Paths to the original YOLOv8 dataset directories
yolo_train_images = os.path.join(base_path, 'original dataset yolov8 format/train/images')
yolo_train_labels = os.path.join(base_path, 'original dataset yolov8 format/train/labels')
yolo_valid_images = os.path.join(base_path, 'original dataset yolov8 format/valid/images')
yolo_valid_labels = os.path.join(base_path, 'original dataset yolov8 format/valid/labels')
yolo_test_images = os.path.join(base_path, 'original dataset yolov8 format/test/images')
yolo_test_labels = os.path.join(base_path, 'original dataset yolov8 format/test/labels')

# Paths to the CSV TensorFlow dataset directories
csv_train_images = os.path.join(base_path, 'csv tensorflow with augmentations/train/images')
csv_valid_images = os.path.join(base_path, 'csv tensorflow with augmentations/valid/images')
csv_test_images = os.path.join(base_path, 'csv tensorflow with augmentations/test/images')

# Output CSV paths
output_train_csv = os.path.join(base_path, 'csv tensorflow with augmentations/train/labels.csv')
output_valid_csv = os.path.join(base_path, 'csv tensorflow with augmentations/valid/labels.csv')
output_test_csv = os.path.join(base_path, 'csv tensorflow with augmentations/test/labels.csv')

# Ensure the directories exist
required_dirs = [
    yolo_train_images, yolo_train_labels, yolo_valid_images, yolo_valid_labels, yolo_test_images, yolo_test_labels,
    csv_train_images, csv_valid_images, csv_test_images,
    os.path.dirname(output_train_csv), os.path.dirname(output_valid_csv), os.path.dirname(output_test_csv)
]

for directory in required_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

print("Starting to generate CSV files...")
print(f"Checking directories and files...")

print(f"YOLO Train Images Path: {yolo_train_images}")
print(f"YOLO Train Labels Path: {yolo_train_labels}")
print(f"CSV Train Images Path: {csv_train_images}")
print(f"Output Train CSV Path: {output_train_csv}")

print(f"YOLO Valid Images Path: {yolo_valid_images}")
print(f"YOLO Valid Labels Path: {yolo_valid_labels}")
print(f"CSV Valid Images Path: {csv_valid_images}")
print(f"Output Valid CSV Path: {output_valid_csv}")

print(f"YOLO Test Images Path: {yolo_test_images}")
print(f"YOLO Test Labels Path: {yolo_test_labels}")
print(f"CSV Test Images Path: {csv_test_images}")
print(f"Output Test CSV Path: {output_test_csv}")

# Generate labels CSV for each dataset
generate_labels_csv(yolo_train_images, yolo_train_labels, output_train_csv)
generate_labels_csv(yolo_valid_images, yolo_valid_labels, output_valid_csv)
generate_labels_csv(yolo_test_images, yolo_test_labels, output_test_csv)

print("Labels CSV generation completed.")
