import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0, VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Concatenate, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.utils import class_weight
import os

# Load preprocessed data
train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
valid_images = np.load('valid_images.npy')
valid_labels = np.load('valid_labels.npy')

# Verify the number of unique labels (emotions) in the dataset
num_classes = len(np.unique(train_labels))

# Convert labels to categorical
train_labels_cat = to_categorical(train_labels, num_classes=num_classes)
valid_labels_cat = to_categorical(valid_labels, num_classes=num_classes)

# Compute class weights to handle imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = dict(enumerate(class_weights))

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(train_images, train_labels_cat, batch_size=32)
valid_generator = valid_datagen.flow(valid_images, valid_labels_cat, batch_size=32)

# Define the ensemble model
def create_ensemble_model():
    # EfficientNetV2B0
    effnet_input = Input(shape=(48, 48, 3))
    effnet_base = EfficientNetV2B0(weights='imagenet', include_top=False, input_tensor=effnet_input)
    effnet_x = GlobalAveragePooling2D()(effnet_base.output)
    effnet_x = BatchNormalization()(effnet_x)
    effnet_x = Dense(256, activation='relu')(effnet_x)
    effnet_x = Dropout(0.5)(effnet_x)

    # VGG19
    vgg_input = Input(shape=(48, 48, 3))
    vgg_base = VGG19(weights='imagenet', include_top=False, input_tensor=vgg_input)
    vgg_x = GlobalAveragePooling2D()(vgg_base.output)
    vgg_x = BatchNormalization()(vgg_x)
    vgg_x = Dense(256, activation='relu')(vgg_x)
    vgg_x = Dropout(0.5)(vgg_x)

    # Concatenate outputs
    concatenated = Concatenate()([effnet_x, vgg_x])
    output = Dense(num_classes, activation='softmax')(concatenated)

    model = Model(inputs=[effnet_input, vgg_input], outputs=output)
    return model

# Instantiate and compile the model
ensemble_model = create_ensemble_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
ensemble_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping, learning rate reduction, and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
model_checkpoint = ModelCheckpoint('best_ensemble_emotion_model.keras', save_best_only=True, monitor='val_loss', mode='min')
tensorboard_callback = TensorBoard(log_dir='logs/fit/ensemble', histogram_freq=1)

# Train the model
history = ensemble_model.fit(
    [train_images, train_images],
    train_labels_cat,
    validation_data=([valid_images, valid_images], valid_labels_cat),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr, model_checkpoint, tensorboard_callback],
    class_weight=class_weights_dict
)

# Save the final model
ensemble_model.save('final_ensemble_emotion_model.keras')
