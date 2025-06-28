from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
from ultralytics import YOLO
import cv2
import os
import uuid
import torch

app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load ML models
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-tc-bible-big-mul-deu_eng_fra_por_spa")
sentiment_analyzer = pipeline("sentiment-analysis")
model = YOLO("yolov8s.pt")

# Allowed file types and max size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate", response_class=HTMLResponse)
async def translate(request: Request, text_input: str = Form(...)):
    try:
        translation = translator(text_input)[0]['translation_text']
        return templates.TemplateResponse("index.html", {
            "request": request,
            "translated_text": translation
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Translation error: {str(e)}"
        })


@app.post("/sentiment", response_class=HTMLResponse)
async def sentiment(request: Request, text_input_sentiment: str = Form(...)):
    try:
        result = sentiment_analyzer(text_input_sentiment)[0]
        return templates.TemplateResponse("index.html", {
            "request": request,
            "sentiment_label": result['label'],
            "sentiment_score": round(result['score'], 4)
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Sentiment analysis error: {str(e)}"
        })


@app.post("/detect", response_class=HTMLResponse)
async def detect_people(request: Request, image: UploadFile = File(...)):
    try:
        # Validate file type and size
        if not allowed_file(image.filename):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "File type not allowed"
            })

        if image.size > MAX_FILE_SIZE:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "File too large. Max 10MB."
            })

        # Save uploaded image
        ext = os.path.splitext(image.filename)[1]
        unique_id = str(uuid.uuid4())
        original_path = f"static/uploaded_{unique_id}{ext}"
        annotated_path = f"static/annotated_{unique_id}{ext}"

        contents = await image.read()
        with open(original_path, "wb") as f:
            f.write(contents)

        # Run YOLO inference
        results = model(original_path)[0]

        # Count people (class 0 is person in COCO)
        people_count = sum(1 for c in results.boxes.cls if int(c) == 0)

        # Annotate image
        annotated_image = results.plot()
        cv2.imwrite(annotated_path, annotated_image)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "people_count": people_count,
            "image_path": f"/{annotated_path}"
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Image processing error: {str(e)}"
        })