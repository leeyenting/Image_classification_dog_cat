# Load vgg16 model and train on custom data to do binary image classification

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
import glob
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# Define the paths to your image dataset and train/test splits
train_dir = "/Data_dogcat/train"
test_dir = "/Data_dogcat/test"
valid_dir = "/Data_dogcat/valid"

# Define the image size and batch size for training and testing
img_size = 224
batch_size = 16
epoch = 5

# Define and load vgg16 model
model = VGG16(include_top=False, input_shape=(224, 224, 3))

# mark loaded layers as not trainable
for layer in model.layers:
    layer.trainable = False

# add new classifier layers
flat1 = tf.keras.layers.Flatten()(model.layers[-1].output)
class1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = tf.keras.layers.Dense(1, activation='sigmoid')(class1)

# define new model
model = tf.keras.models.Model(inputs=model.inputs, outputs=output)

# compile model
opt = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


checkpoint_path = "model_checkpoint_dogcat.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, 
    save_best_only=True, 
    save_weights_only=False, 
    verbose=1
)


# Create a data generator for the training data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True)

# specify imagenet mean values for centering
datagen.mean = [123.68, 116.779, 103.939]


# Load the training data
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

# Load the validationg data
valid_generator = datagen.flow_from_directory(
    valid_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

# Load the testing data
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)


# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epoch,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size
)

# Evaluate the model on the test data
print(test_generator.samples // batch_size)
test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.samples // batch_size)
print("Test accuracy:", test_acc)



