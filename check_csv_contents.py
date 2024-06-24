import pandas as pd
import os

# Base path for the dataset
base_path = 'D:/EmotionDetector/dataset'

# Paths to the labels CSV files
train_labels_csv = os.path.join(base_path, 'csv tensorflow with augmentations/train/labels.csv')
valid_labels_csv = os.path.join(base_path, 'csv tensorflow with augmentations/valid/labels.csv')
test_labels_csv = os.path.join(base_path, 'csv tensorflow with augmentations/test/labels.csv')

# Function to check CSV content
def check_csv(path):
    try:
        df = pd.read_csv(path)
        print(f"CSV at {path} loaded successfully with {len(df)} entries.")
        print(df.head())
    except Exception as e:
        print(f"Error loading CSV at {path}: {e}")

check_csv(train_labels_csv)
check_csv(valid_labels_csv)
check_csv(test_labels_csv)
