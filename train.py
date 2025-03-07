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

# logging
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
GITHUB_REPO = 'zh3nru/model_CI' 
GITHUB_MODEL_PATH = 'data/models' 
MY_TOKEN = os.getenv('MY_TOKEN')

if not MY_TOKEN:
    logging.critical("GitHub token not found in environment variables. No 'MY_TOKEN'.")
    sys.exit(1)

def get_github_file(repo_name, file_path, github_token):
    url = f"{GITHUB_API_URL}/repos/{repo_name}/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_info = response.json()
        content = file_info['content']
        sha = file_info['sha']
        logging.info(f"Successfully fetched {file_path} from {repo_name}.")
        return content, sha
    else:
        return None, None

def upload_file_to_github(repo_name, file_path, file_content, github_token, commit_message="Upload model file"):
    url = f"{GITHUB_API_URL}/repos/{repo_name}/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Check if the file already exists
    existing_content, existing_sha = get_github_file(repo_name, file_path, github_token)

    data = {
        "message": commit_message,
        "content": b64encode(file_content).decode('utf-8'),
        "branch": "main"  
    }

    if existing_sha:
        data["sha"] = existing_sha

    response = requests.put(url, headers=headers, data=json.dumps(data))

    if response.status_code in [200, 201]:
        action = "Updated" if existing_sha else "Uploaded"
        logging.info(f"{action} {file_path} in {repo_name} successfully.")
        return True
    else:
        return False

emotions = ["Aversion", "Anger", "Happiness", "Fear", "Sadness", "Surprise", "Peace"]

# Path for training and validation data
train_data_path = Path(os.getenv('TRAIN_DATA_PATH', 'data/train_gen_frames'))
val_data_path = Path(os.getenv('VAL_DATA_PATH', 'data/train_gen_frames'))
updated_model_path = Path(os.getenv('UPDATED_MODEL_PATH', 'data/models'))

updated_model_path.mkdir(parents=True, exist_ok=True)

existing_model_file = os.getenv('EXISTING_MODEL_FILE', 'eMotion.h5')  
existing_model_path = updated_model_path / existing_model_file

# Get current date
current_date = dt.datetime.now().strftime('%Y%m%d')

# Add date to main and tflite model file name
updated_model_file = f'updated_model.keras' 
updated_model_save_path = updated_model_path / updated_model_file

tflite_model_file = f'updated_model_{current_date}.tflite'
tflite_model_save_path = updated_model_path / tflite_model_file

# Data augmentation
train_data_aug = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10, 
    zoom_range=0.1,
    validation_split=0.2
)

validation_data_aug = ImageDataGenerator(rescale=1./255)

try:
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
    # Load current model
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

    # Convert main model to tflite model
    converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_model_path = updated_model_save_path.with_suffix('.tflite')

    # Save tflite model
    with open(tflite_model_save_path, 'wb') as f:
        f.write(tflite_model)
    logging.info(f"TensorFlow Lite model saved to {tflite_model_save_path}")

    for layer in emotion_model.layers:
        if hasattr(layer, 'get_weights') and hasattr(layer, 'set_weights'):
            weights = layer.get_weights()
            if weights:
                weights = [w.astype('float16') for w in weights]
                layer.set_weights(weights)

    # Save main model
    emotion_model.save(
        str(updated_model_save_path),
        save_format='keras',
        include_optimizer=False
    )
    logging.info(f"Updated Keras model saved to {updated_model_save_path}")

    # Second repository path where tflite model will be sent
    SECOND_GITHUB_REPO = 'Kristoferseyan/fix-emotion'  
    SECOND_GITHUB_MODEL_PATH = 'assets/models'  

    def upload_file_to_second_github_repo(repo_name, file_path, file_content, github_token, commit_message="Upload model file to second repo"):
        url = f"{GITHUB_API_URL}/repos/{repo_name}/contents/{file_path}"
        headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
        }

        # Check if file already exists
        existing_content, existing_sha = get_github_file(repo_name, file_path, github_token)

        data = {
            "message": commit_message,
            "content": b64encode(file_content).decode('utf-8'),
            "branch": "main"  
        }

        if existing_sha:
            data["sha"] = existing_sha

        response = requests.put(url, headers=headers, data=json.dumps(data))

        if response.status_code in [200, 201]:
            action = "Updated" if existing_sha else "Uploaded"
            logging.info(f"{action} {file_path} in {repo_name} successfully.")
            return True
        else:
            return False

    def upload_models():
        models_to_upload = {
        'keras': updated_model_save_path,           
        'tflite': tflite_model_save_path
        }

        for model_type, model_path in models_to_upload.items():
            if model_path.exists():
                with open(model_path, 'rb') as file:
                    file_content = file.read()

                # Upload main model to first repository
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
                    logging.info(f"Successfully uploaded {model_path.name} to first GitHub repository.")
                
                # Upload tflite model to second repository
                second_github_file_path = f"{SECOND_GITHUB_MODEL_PATH}/{model_path.name}"
                second_commit_msg = f"Upload updated {model_type} model to second repo: {model_path.name}"
                success_second = upload_file_to_second_github_repo(
                    repo_name=SECOND_GITHUB_REPO,
                    file_path=second_github_file_path,
                    file_content=file_content,
                    github_token=MY_TOKEN,
                    commit_message=second_commit_msg
                )
                if success_second:
                    logging.info(f"Successfully uploaded {model_path.name} to second GitHub repository.")
                else:
                    logging.error(f"Failed to upload {model_path.name} to second GitHub repository.")
            else:
                logging.warning(f"Model file {model_path} does not exist and cannot be uploaded.")

    upload_models()


except Exception as e:
    logging.error(f"Training failed: {e}")
    sys.exit(1)
