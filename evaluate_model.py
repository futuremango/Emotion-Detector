import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load test data
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

# Convert labels to categorical
num_classes = len(np.unique(test_labels))
test_labels_cat = to_categorical(test_labels, num_classes=num_classes)

# Normalize test images
test_images = test_images / 255.0

# Load the ensemble model
ensemble_model = load_model('final_ensemble_emotion_model.keras')

# Evaluate the model
loss, accuracy = ensemble_model.evaluate([test_images, test_images], test_labels_cat)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Generate predictions
predictions = ensemble_model.predict([test_images, test_images])
predicted_labels = np.argmax(predictions, axis=1)

# Classification report
print(classification_report(test_labels, predicted_labels, zero_division=1))

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
