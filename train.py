import os
from pathlib import Path
import sys
import logging

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model
from tqdm import tqdm
import datetime as dt
import imageio
from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define emotion categories
emotions = ["Aversion", "Anger", "Happiness", "Fear", "Sadness", "Surprise", "Peace"]

# Set paths using environment variables or default values
train_data_path = Path(os.getenv('train_data_path', 'data/train_gen_frames'))
val_data_path = Path(os.getenv('val_data_path', 'data/train_gen_frames'))
updated_model_path = Path(os.getenv('updated_model_path', 'data/models'))

# Ensure the model directory exists
updated_model_path.mkdir(parents=True, exist_ok=True)

# Define paths for existing and updated models
existing_model_file = Path(os.getenv('existing_model_file', 'eNotion.h5'))
existing_model_path = updated_model_path / existing_model_file

# Define the path for the updated model
updated_model_file = Path(os.getenv('updated_model_file', 'updated_model.h5'))
updated_model_save_path = updated_model_path / updated_model_file

# Initialize ImageDataGenerators for training and validation
train_data_aug = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,  # Changed from 0.1 to 10 degrees for rotation
    zoom_range=0.1,
    validation_split=0.2
)

validation_data_aug = ImageDataGenerator(rescale=1./255)

try:
    # Load training data
    train_data = train_data_aug.flow_from_directory(
        str(train_data_path),
        target_size=(64, 64),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='training'
    )

    # Load validation data
    val_data = train_data_aug.flow_from_directory(
        str(val_data_path),
        target_size=(64, 64),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    logging.info(f"Shape of train images: {train_data.image_shape}")
    logging.info(f"Number of training samples: {train_data.samples}")
    logging.info(f"Number of training classes: {len(train_data.class_indices)}")
    logging.info(f"Number of validation samples: {val_data.samples}")

except Exception as e:
    logging.error(f"Error loading images: {e}")
    sys.exit(1)

try:
    # Load the existing model
    if existing_model_path.exists():
        logging.info(f"Loading existing model from {existing_model_path}")
        emotion_model = load_model(str(existing_model_path))
        logging.info("Model loaded successfully.")
    else:
        logging.error(f"Existing model file not found at {existing_model_path}")
        sys.exit(1)

    # Optional: Print model summary
    emotion_model.summary()

    # Compile the model (necessary if you changed trainable layers)
    emotion_model.compile(
        optimizer=Adam(learning_rate=1e-4),  # You can adjust the learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    checkpoint = ModelCheckpoint(
        filepath=str(updated_model_save_path),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # Train the model
    history = emotion_model.fit(
        train_data,
        steps_per_epoch=train_data.samples // train_data.batch_size,
        epochs=20,
        validation_data=val_data,
        validation_steps=val_data.samples // val_data.batch_size,
        callbacks=[early_stopping, checkpoint]
    )

    # Save the updated model
    emotion_model.save(str(updated_model_save_path))
    logging.info(f"Updated model saved to {updated_model_save_path}")

except Exception as e:
    logging.error(f"Training failed: {e}")
    sys.exit(1)
