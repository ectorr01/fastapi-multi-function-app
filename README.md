# FastAPI Multi-Function Web App

A simple web application built with **FastAPI** that provides multiple AI-powered features:

- Text translation (English → French)
- Sentiment analysis
- Person detection in images using YOLOv8

The app uses modern ML models and is served through a Jinja2 HTML template interface.

---

## 🧠 Features

1. **Translation**: Translate English text to French using Hugging Face Transformers.
2. **Sentiment Analysis**: Analyze the sentiment of any given text.
3. **Object Detection**: Detect people in uploaded images using YOLOv8 from Ultralytics.

---

## 🛠️ Technologies Used

| Library | Purpose |
|--------|---------|
| [FastAPI](https://fastapi.tiangolo.com/) | Backend framework for building APIs |
| [Transformers](https://huggingface.co/docs/transformers/) | For NLP tasks like translation and sentiment analysis |
| [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/) | For real-time object detection in images |
| [Torch (PyTorch)](https://pytorch.org/) | Machine learning backend used by both Transformers and YOLO |
| [OpenCV](https://opencv.org/) | Image manipulation and saving annotated results |
| [Jinja2](https://jinja.palletsprojects.com/) | HTML templating engine |
| [Uvicorn](https://www.starlette.io/) | ASGI server to run the FastAPI app |

---

## 🚀 How to Run the App

### 1. Clone the repository:
```bash
git clone https://github.com/ectorr01/fastapi-multi-function-app.git
cd fastapi-multi-function-app
```

### 2. (Optional) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the app:
```bash
uvicorn main:app --reload
```

### 5. Open in browser:
Go to: http://localhost:8000

---

## 📁 Project Structure

```
fastapi-multi-function-app/
├── main.py
├── requirements.txt
├── templates/
│   └── index.html
└── static/
```

---

## ✅ Requirements

Make sure you have installed:

- Python 3.8+
- FastAPI
- Uvicorn
- Transformers
- Torch
- Ultralytics (YOLO)
- OpenCV (`opencv-python-headless`)
- Jinja2

All required packages are listed in `requirements.txt`.

---
