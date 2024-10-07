import os
from pathlib import Path
import sys
import logging
import requests
import base64
from dotenv import load_dotenv

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

# Load environment variables from a .env file if present
load_dotenv()

# GitHub Configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_REPO = os.getenv('GITHUB_REPO', 'zh3nru/model_CI')
GITHUB_BRANCH = os.getenv('GITHUB_BRANCH', 'main')
GITHUB_TARGET_FOLDER = os.getenv('GITHUB_TARGET_FOLDER', 'data/models')

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# Emotion Categories
emotions = ["Aversion", "Anger", "Happiness", "Fear", "Sadness", "Surprise", "Peace"]

# Data Paths
train_data_path = Path(os.getenv('train_data_path', 'data/train_gen_frames'))
val_data_path = Path(os.getenv('val_data_path', 'data/train_gen_frames'))
updated_model_path = Path(os.getenv('updated_model_path', 'data/models'))

# Ensure the models directory exists
updated_model_path.mkdir(parents=True, exist_ok=True)

# Model File Paths
existing_model_file = os.getenv('existing_model_file', 'eMotion.h5')
existing_model_path = updated_model_path / existing_model_file

updated_model_file = 'updated_model.keras'
updated_model_save_path = updated_model_path / updated_model_file

# Image Data Generators
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

def upload_file_to_github(file_path, repo, folder, branch, github_token, commit_message):
    """
    Uploads or updates a file in the specified GitHub repository and folder.
    
    :param file_path: Path to the local file to upload.
    :param repo: GitHub repository in the format 'owner/repo'.
    :param folder: Target folder in the repository.
    :param branch: Branch to commit to.
    :param github_token: GitHub Personal Access Token.
    :param commit_message: Commit message for the upload.
    """
    try:
        with open(file_path, "rb") as file:
            content = file.read()
        encoded_content = base64.b64encode(content).decode('utf-8')
        filename = os.path.basename(file_path)
        github_api_url = f"https://api.github.com/repos/{repo}/contents/{folder}/{filename}"
        
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Check if the file already exists to get its SHA
        get_response = requests.get(github_api_url, headers=headers)
        if get_response.status_code == 200:
            sha = get_response.json()['sha']
            logging.info(f"File {filename} exists. Updating the file.")
        elif get_response.status_code == 404:
            sha = None
            logging.info(f"File {filename} does not exist. Creating a new file.")
        else:
            logging.error(f"Failed to check existence of {filename} on GitHub. Status code: {get_response.status_code}. Response: {get_response.json()}")
            return False
        
        data = {
            "message": commit_message,
            "content": encoded_content,
            "branch": branch
        }
        
        if sha:
            data["sha"] = sha
        
        put_response = requests.put(github_api_url, headers=headers, json=data)
        
        if put_response.status_code in [200, 201]:
            logging.info(f"Successfully uploaded {filename} to GitHub repository {repo} in {folder}/.")
            return True
        else:
            logging.error(f"Failed to upload {filename} to GitHub. Status code: {put_response.status_code}. Response: {put_response.json()}")
            return False
    except Exception as e:
        logging.error(f"Exception occurred while uploading {file_path} to GitHub: {e}")
        return False

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

    # Convert Keras model to TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Define TFLite model path
    tflite_model_file = 'updated_model.tflite'
    tflite_model_save_path = updated_model_path / tflite_model_file

    # Save the TensorFlow Lite model locally
    with open(tflite_model_save_path, 'wb') as f:
        f.write(tflite_model)
    logging.info(f"TensorFlow Lite model saved to {tflite_model_save_path}")

    # Save the updated Keras model locally
    emotion_model.save(str(updated_model_save_path))
    logging.info(f"Updated Keras model saved to {updated_model_save_path}")

    # Upload the updated Keras model to GitHub
    if GITHUB_TOKEN:
        commit_msg_keras = f"Update Keras model: {updated_model_file} at {dt.datetime.now().isoformat()}"
        success_keras = upload_file_to_github(
            file_path=updated_model_save_path,
            repo=GITHUB_REPO,
            folder=GITHUB_TARGET_FOLDER,
            branch=GITHUB_BRANCH,
            github_token=GITHUB_TOKEN,
            commit_message=commit_msg_keras
        )
        
        # Upload the TensorFlow Lite model to GitHub
        commit_msg_tflite = f"Update TFLite model: {tflite_model_file} at {dt.datetime.now().isoformat()}"
        success_tflite = upload_file_to_github(
            file_path=tflite_model_save_path,
            repo=GITHUB_REPO,
            folder=GITHUB_TARGET_FOLDER,
            branch=GITHUB_BRANCH,
            github_token=GITHUB_TOKEN,
            commit_message=commit_msg_tflite
        )
        
        if success_keras and success_tflite:
            logging.info("Both models uploaded to GitHub successfully.")
        else:
            logging.error("One or both models failed to upload to GitHub.")
    else:
        logging.warning("GITHUB_TOKEN not found. Skipping GitHub upload.")

except Exception as e:
    logging.error(f"Training failed: {e}")
    sys.exit(1)
