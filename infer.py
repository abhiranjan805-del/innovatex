from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import json
import cv2
import os
import requests
from bs4 import BeautifulSoup

# -----------------------------
# Initialize App
# -----------------------------
app = FastAPI(title="AI Crop Disease Detection API")

# -----------------------------
# Enable CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "saved_models/crop_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "saved_models/label_map.npy")
DATA_JSON_PATH = os.path.join(BASE_DIR, "data.json")

# -----------------------------
# Load Model & Data
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
index_to_label = {v: k for k, v in label_map.items()}

with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
    DATA = json.load(f)

print("✅ Model & Dataset Loaded")

# -----------------------------
# 🔍 Web Search Function
# -----------------------------
def search_disease_info(disease_name):
    try:
        query = f"{disease_name} crop disease cause prevention cure"
        url = f"https://duckduckgo.com/html/?q={query}"

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)

        soup = BeautifulSoup(response.text, "html.parser")

        results = soup.find_all("a", class_="result__a", limit=6)
        texts = [r.get_text() for r in results]

        return {
            "description": texts[0] if len(texts) > 0 else "",
            "cause": texts[1] if len(texts) > 1 else "",
            "prevention": texts[2] if len(texts) > 2 else "",
            "cure": texts[3] if len(texts) > 3 else "",
            "suggestion": texts[4] if len(texts) > 4 else "",
        }

    except Exception:
        return None

# -----------------------------
# 🔁 Dataset Fallback
# -----------------------------
def normalize(text):
    return text.lower().replace("_", "").replace(" ", "")

def find_dataset_match(predicted_class):
    pred = normalize(predicted_class)

    for key in DATA.keys():
        if pred in normalize(key) or normalize(key) in pred:
            return key

    return None

# -----------------------------
# 🚀 MAIN API
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # -----------------------------
        # Read Image
        # -----------------------------
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"error": "Invalid image"})

        # -----------------------------
        # Preprocess
        # -----------------------------
        img = cv2.resize(img, (224, 224)) / 255.0
        img = np.expand_dims(img, axis=0)

        # -----------------------------
        # Predict
        # -----------------------------
        prediction = model.predict(img)[0]
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        disease = index_to_label[class_index]

        # -----------------------------
        # 🔥 Step 1: Web Search
        # -----------------------------
        search_name = disease.replace("___", " ")
        web_info = search_disease_info(search_name)

        # -----------------------------
        # 🔁 Step 2: Dataset Fallback
        # -----------------------------
        dataset_info = {}
        matched_key = find_dataset_match(disease)

        if matched_key:
            dataset_info = DATA.get(matched_key, {})

        # -----------------------------
        # 🧠 Merge Logic (Web + Dataset)
        # -----------------------------
        final_info = {
            "description": "",
            "cause": "",
            "prevention": "",
            "cure": "",
            "suggestion": ""
        }

        for key in final_info.keys():
            # Priority: Web → Dataset
            if web_info and web_info.get(key):
                final_info[key] = web_info.get(key)
            elif dataset_info.get(key):
                final_info[key] = dataset_info.get(key)

        # -----------------------------
        # Final Response
        # -----------------------------
        return {
            "disease": disease,
            "confidence": round(confidence * 100, 2),
            "description": final_info["description"],
            "cause": final_info["cause"],
            "prevention": final_info["prevention"],
            "cure": final_info["cure"],
            "suggestion": final_info["suggestion"]
        }

    except Exception as e:
        return JSONResponse({"error": str(e)})

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)