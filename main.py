import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

# --- CONFIG ---
API_KEY = "NVfp8h9atJEAWzsw1eZ0"   # replace with your Roboflow API key

# Dictionary of plant -> Roboflow model IDs
MODELS = {
    "rice": "rice-plant-leaf-disease-classification/1",
    "cassava": "cassava-leaf-disease-classification/1",
    "sugarcane": "sugarcane-leaf-disease-classification/1",
    "tea": "tea-leaf-plant-diseases/1",
    "mango": "mango-leaf-disease-2/4"
}

app = FastAPI(title="Plant Disease Classification API")

def classify_image(image_file, plant_type: str):
    """
    Send image to Roboflow classification API based on plant type.
    """
    if plant_type not in MODELS:
        return {"error": f"Invalid plant type. Choose from {list(MODELS.keys())}"}

    model_id = MODELS[plant_type]
    api_url = f"https://classify.roboflow.com/{model_id}?api_key={API_KEY}"

    response = requests.post(api_url, files={"file": image_file})

    if response.status_code != 200:
        return {"error": response.text}

    result = response.json()

    if "predictions" in result and len(result["predictions"]) > 0:
        top_pred = result["predictions"][0]
        disease = top_pred["class"]
        confidence = round(top_pred["confidence"] * 100, 2)
        return {
            "plant": plant_type,
            "disease": disease,
            "confidence": confidence,
            "raw_result": result
        }
    else:
        return {"plant": plant_type, "disease": "None", "confidence": 0}

@app.post("/classify")
async def classify(
    plant_type: str = Form(...), 
    file: UploadFile = File(...)
):
    """
    Upload an image + specify plant_type (rice/cassava/sugarcane/tea).
    """
    try:
        image_bytes = await file.read()
        result = classify_image(image_bytes, plant_type)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
