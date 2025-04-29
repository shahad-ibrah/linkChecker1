# main.py
import os
import joblib
import json
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, HttpUrl, Field
from contextlib import asynccontextmanager
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration (Should match training artifacts location) ---
ARTIFACTS_DIR = 'pytorch_model_artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'url_predictor_model.pth')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler_pytorch.joblib')
TLD_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'tld_encoder_pytorch.joblib')
CHAR_MAP_PATH = os.path.join(ARTIFACTS_DIR, 'char_map_pytorch.json')
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, 'model_config.json')

ml_models = {}

class URLClassifier(nn.Module):
    def __init__(self, vocab_size, char_embedding_dim, lstm_hidden_dim,
                 num_tld_classes, tld_embedding_dim, num_numerical_features,
                 combined_features_dim, dropout_rate=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.char_embedding_dim = char_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_tld_classes = num_tld_classes
        self.tld_embedding_dim = tld_embedding_dim
        self.num_numerical_features = num_numerical_features
        self.combined_features_dim = combined_features_dim
        self.dropout_rate = dropout_rate
        self.char_embedding = nn.Embedding(vocab_size, char_embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(char_embedding_dim, lstm_hidden_dim, batch_first=True)
        self.tld_embedding = nn.Embedding(num_tld_classes, tld_embedding_dim)
        total_combined_dim = lstm_hidden_dim + tld_embedding_dim + num_numerical_features
        self.fc1 = nn.Linear(total_combined_dim, combined_features_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(combined_features_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, url_indices, numerical_features, tld_indices):
        char_embeds = self.char_embedding(url_indices)
        _, (hidden, _) = self.lstm(char_embeds)
        url_features = hidden[-1]
        tld_embeds = self.tld_embedding(tld_indices)
        combined = torch.cat((url_features, numerical_features, tld_embeds), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output

import re
from urllib.parse import urlparse
import ipaddress

def get_tld(url):
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        if hostname:
            parts = hostname.split('.')
            if len(parts) > 1 and parts[-1]:
                 tld = parts[-1]
                 if not tld.isnumeric() and len(tld) > 1:
                      return tld
        return ""
    except Exception:
        return ""

def is_ip_address(url):
    try:
        hostname = urlparse(url).hostname
        if hostname:
            ipaddress.ip_address(hostname)
            return 1
        return 0
    except ValueError:
        return 0
    except Exception:
        return 0

def extract_url_features(url: str):
    if not isinstance(url, str):
         raise ValueError("Input URL must be a string.")
    url_length = len(url)
    is_domain_ip = is_ip_address(url)
    tld = get_tld(url) if is_domain_ip == 0 else ""
    return {
        'URL': url,
        'URLLength': url_length,
        'IsDomainIP': is_domain_ip,
        'TLD': tld
    }


# --- Loading function for artifacts (Same as before) ---
def load_artifacts():
    print("Loading model and preprocessor artifacts...")
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, TLD_ENCODER_PATH, CHAR_MAP_PATH, CONFIG_PATH]):
        raise FileNotFoundError(f"One or more artifact files not found in {ARTIFACTS_DIR}. Please ensure training was completed and artifacts saved.")
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        ml_models['config'] = config
    ml_models['scaler'] = joblib.load(SCALER_PATH)
    ml_models['tld_encoder'] = joblib.load(TLD_ENCODER_PATH)
    with open(CHAR_MAP_PATH, 'r') as f:
        char_map_data = json.load(f)
        ml_models['char_to_idx'] = char_map_data['char_to_idx']
        if config['vocab_size'] != char_map_data['VOCAB_SIZE']:
             print(f"Warning: Config vocab size {config['vocab_size']} differs from loaded char map size {char_map_data['VOCAB_SIZE']}.")
             config['vocab_size'] = char_map_data['VOCAB_SIZE']
    model = URLClassifier(
        vocab_size=config['vocab_size'], char_embedding_dim=config['char_embedding_dim'],
        lstm_hidden_dim=config['lstm_hidden_dim'], num_tld_classes=config['num_tld_classes'],
        tld_embedding_dim=config['tld_embedding_dim'], num_numerical_features=config['num_numerical_features'],
        combined_features_dim=config['combined_features_dim'], dropout_rate=config['dropout_rate']
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    ml_models['model'] = model
    ml_models['device'] = device
    print(f"Artifacts loaded successfully. Model running on device: {device}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    ml_models.clear()
    print("ML models cleared.")

app = FastAPI(title="URL Safety Predictor", lifespan=lifespan)
# Enable CORS for frontend interaction (e.g., Streamlit, React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify a list like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLInput(BaseModel):
    url: str = Field(..., min_length=10, example="https://www.google.com")

class PredictionResponse(BaseModel):
    url: str
    is_safe: bool = Field(..., description="True if the URL is predicted as safe, False otherwise")
    probability_safe: float = Field(..., description="Model's confidence score (probability of being safe [label 1])")
    status: str = Field(..., description="Indicates how the prediction was determined") # Changed default

class PredictionErrorResponse(BaseModel):
    url: str
    status: str
    detail: str

async def run_prediction(url: str):
    model = ml_models.get('model')
    scaler = ml_models.get('scaler')
    tld_encoder = ml_models.get('tld_encoder')
    char_to_idx = ml_models.get('char_to_idx')
    config = ml_models.get('config')
    device = ml_models.get('device')

    if not all([model, scaler, tld_encoder, char_to_idx, config, device]):
        raise HTTPException(status_code=500, detail="Model or preprocessors not loaded correctly.")

    try:
        features = await run_in_threadpool(extract_url_features, url)
        numerical_data = pd.DataFrame([features], columns=config['numerical_cols'])
        scaled_numerical = await run_in_threadpool(scaler.transform, numerical_data)
        num_tensor = torch.tensor(scaled_numerical, dtype=torch.float32).to(device)

        tld = features['TLD']
        try:
            tld_encoded_array = await run_in_threadpool(tld_encoder.transform, [tld])
            tld_idx = tld_encoded_array[0]
            tld_tensor = torch.tensor([tld_idx], dtype=torch.long).to(device)

        except ValueError:
            print(f"Warning: TLD '{tld}' not found in encoder for URL '{url}'. Classifying as not safe.")
            return 0, 0.0, "Classified as not safe (preprocessing issue: unknown TLD)"

        max_len = config['max_url_len']
        unknown_char_idx = 0
        url_indices = [char_to_idx.get(char, unknown_char_idx) for char in url]
        padded_url = np.zeros(max_len, dtype=np.int64)
        seq_len = min(len(url_indices), max_len)
        padded_url[:seq_len] = url_indices[:seq_len]
        url_tensor = torch.tensor(padded_url, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
             output = model(url_tensor, num_tensor, tld_tensor)

        probability = output.item()
        predicted_label = 1 if probability > 0.5 else 0

        return predicted_label, probability, "Prediction successful"

    except Exception as e:
        print(f"Error during prediction pipeline for URL '{url}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction pipeline: {str(e)}")


# --- API Endpoint ---
@app.post("/predict/",
          response_model=PredictionResponse,
          responses={500: {"model": PredictionErrorResponse}}) 
async def predict_url_safety(payload: URLInput):
    """
    Predicts if a given URL is safe (returns true) or not (returns false).
    Handles URLs with unknown TLDs (including IP addresses) by classifying them as not safe.

    - **url**: The URL string to analyze.
    """
    input_url = payload.url
    try:
        predicted_label, probability, status_message = await run_prediction(input_url)

        is_safe_boolean = (predicted_label == 1)

        return PredictionResponse(
            url=input_url,
            is_safe=is_safe_boolean,
            probability_safe=probability,
            status=status_message 
        )

    except HTTPException as e:

         raise e
    except Exception as e:
        print(f"Unexpected error in /predict endpoint for URL '{input_url}': {e}")

        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")


@app.get("/", summary="Root endpoint", description="Provides a simple welcome message.")
async def read_root():
    return {"message": "Welcome to the URL Safety Predictor API. Use the /predict/ endpoint to analyze URLs."}

# uvicorn main:app --reload --host 0.0.0.0 --port 8000