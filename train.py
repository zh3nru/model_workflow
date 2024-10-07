import os
from pathlib import Path
import sys
import logging
import json
from urllib.parse import urlparse, unquote
from base64 import b64encode

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
import requests

import tensorflow_model_optimization as tfmot  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# GitHub repository details
GITHUB_API_URL = "https://api.github.com"
GITHUB_REPO = 'zh3nru/model_CI'  # Repository in the format 'owner/repo'
GITHUB_MODEL_PATH = 'data/models'  # Path within the repository to save models
MY_TOKEN = os.getenv('MY_TOKEN')

if not MY_TOKEN:
    logging.critical("GitHub token not found in environment variables. Please set 'MY_TOKEN'.")
    sys.exit(1)

# [Existing functions: get_github_file, upload_file_to_github]

# Define emotions
emotions = ["Aversion", "Anger", "Happiness", "Fear", "Sadness", "Surprise", "Peace"]

# Paths for training and validation data
train_data_path = Path(os.getenv('TRAIN_DATA_PATH', 'data/train_gen_frames'))
val_data_path = Path(os.getenv('VAL_DATA_PATH', 'data/train_gen_frames'))
updated_model_path = Path(os.getenv('UPDATED_MODEL_PATH', 'data/models'))

# Create the directory if it doesn't exist
updated_model_path.mkdir(parents=True, exist_ok=True)

# Change the default existing model file to use .keras extension
existing_model_file = os.getenv('EXISTING_MODEL_FILE', 'eMotion.h5')  # Changed from 'eMotion.h5' to 'eMotion.keras'
existing_model_path = updated_model_path / existing_model_file

# Get current date string
current_date = dt.datetime.now().strftime('%Y%m%d')

# Define updated model filenames with date and .keras extension
updated_model_file = f'updated_model_{current_date}.keras'  # Changed from .h5 to .keras
updated_model_save_path = updated_model_path / updated_model_file

# Define TFLite model filename with date and .tflite extension
tflite_model_file = f'updated_model_{current_date}.tflite'
tflite_model_save_path = updated_model_path / tflite_model_file

# Image data generators with data augmentation for training and rescaling for validation
train_data_aug = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10, 
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

    emotion_model.summary()

    # Compile the model
    emotion_model.compile(
        optimizer=Adam(learning_rate=1e-4), 
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

    # Apply post-training quantization to the Keras model
    def apply_quantization_to_keras_model(model):
        """
        Applies post-training quantization to a Keras model.

        Args:
            model (tf.keras.Model): The original Keras model.

        Returns:
            tf.keras.Model: The quantized Keras model.
        """
        # Apply quantization
        quantized_model = tfmot.quantization.keras.quantize_model(model)

        # Compile the quantized model
        quantized_model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return quantized_model

    # Apply quantization
    quantized_emotion_model = apply_quantization_to_keras_model(emotion_model)

    # Fine-tune the quantized model (optional but recommended)
    logging.info("Starting fine-tuning of the quantized model.")
    history_quantized = quantized_emotion_model.fit(
        train_data,
        steps_per_epoch=train_data.samples // train_data.batch_size,
        epochs=5,  # Adjust the number of epochs as needed
        validation_data=val_data,
        validation_steps=val_data.samples // val_data.batch_size,
        callbacks=[early_stopping, checkpoint]
    )
    logging.info("Fine-tuning of the quantized model completed.")

    # Save the quantized Keras model
    quantized_model_file = f'quantized_model_{current_date}.keras'
    quantized_model_save_path = updated_model_path / quantized_model_file
    quantized_emotion_model.save(str(quantized_model_save_path), save_format='tf')
    logging.info(f"Quantized Keras model saved to {quantized_model_save_path}")

    # Convert Keras model to TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the converted TensorFlow Lite model
    with open(tflite_model_save_path, 'wb') as f:
        f.write(tflite_model)
    logging.info(f"TensorFlow Lite model saved to {tflite_model_save_path}")

    # Save the updated Keras model with .keras extension and explicit format
    emotion_model.save(str(updated_model_save_path), save_format='tf')  # Changed save_format to 'tf'
    logging.info(f"Updated Keras model saved to {updated_model_save_path}")

    def upload_models():
        """
        Uploads the saved model files to the specified GitHub repository.
        """
        models_to_upload = {
            'keras': updated_model_save_path,
            'quantized_keras': quantized_model_save_path,  # Add quantized model
            'tflite': tflite_model_save_path
        }

        for model_type, model_path in models_to_upload.items():
            if model_path.exists():
                with open(model_path, 'rb') as file:
                    file_content = file.read()
                github_file_path = f"{GITHUB_MODEL_PATH}/{model_path.name}"
                commit_msg = f"Upload updated {model_type} model: {model_path.name}"
                success = upload_file_to_github(
                    repo_name=GITHUB_REPO,
                    file_path=github_file_path,
                    file_content=file_content,
                    github_token=MY_TOKEN,
                    commit_message=commit_msg
                )
                if success:
                    logging.info(f"Successfully uploaded {model_path.name} to GitHub.")
                else:
                    logging.error(f"Failed to upload {model_path.name} to GitHub.")
            else:
                logging.warning(f"Model file {model_path} does not exist and cannot be uploaded.")

    # Call the function to upload models to GitHub
    upload_models()

except Exception as e:
    logging.error(f"Training failed: {e}")
    sys.exit(1)
