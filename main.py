from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import requests
import shutil
import os
import cv2
import uuid
from typing import Dict
from starlette.responses import FileResponse

# ------------------- CONFIG -------------------
API_KEY = os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY")  # store in Render settings

# Plant → model mapping
MODEL_MAP: Dict[str, str] = {
    "rice": "rice-plant-leaf-disease-classification/1",
    "cassava": "cassava-model/1",
    "sugarcane": "sugarcane-leaf-disease/2",
    "tea": "tea-leaf-plant-diseases/1"
}

# ------------------- FASTAPI APP -------------------
app = FastAPI()

# Allow Flutter or Postman
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def infer(image_path: str, model_id: str, api_key: str):
    """Call Roboflow API directly using requests"""
    url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}"
    with open(image_path, "rb") as f:
        response = requests.post(url, files={"file": f})
    return response.json()


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    plant: str = Form(...)
):
    """Upload image + plant type → get prediction + annotated image"""

    if plant not in MODEL_MAP:
        return {"error": f"Invalid plant type. Choose from {list(MODEL_MAP.keys())}"}

    # Save uploaded image temporarily
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run inference
    result = infer(temp_filename, MODEL_MAP[plant], API_KEY)

    # Load image for drawing
    image = cv2.imread(temp_filename)

    # Draw bounding boxes
    if "predictions" in result:
        for pred in result["predictions"]:
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            label = pred["class"]
            confidence = pred["confidence"]

            # Coordinates
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2

            # Draw box + label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({confidence:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # Save annotated image
    annotated_filename = f"annotated_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(annotated_filename, image)

    # Delete temp file
    os.remove(temp_filename)

    return {
        "results": result,
        "annotated_image_url": f"/download/{annotated_filename}"
    }


@app.get("/download/{filename}")
async def download(filename: str):
    """Download annotated image"""
    return FileResponse(filename)
