# PraneethSigns - Backend

Flask API that serves predictions from your trained sign_model.keras
CNN to the React frontend.

## 1. Add your trained model files

Copy these two files (from your training project folder) into this
same backend folder:

```
backend/
├── app.py
├── requirements.txt
├── sign_model.keras       <- copy this in
└── class_labels.json      <- copy this in
```

## 2. Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 3. Run

```bash
python app.py
```

You should see:
```
Loading model...
Model loaded. 29 classes: ['A', 'B', 'C', ...]
 * Running on http://127.0.0.1:5000
```

Leave this running - it's your API server. Your React frontend
(running separately via `npm run dev`) will call this at
`http://localhost:5000/api/predict`.

## 4. Test it without the frontend

You can confirm the backend works on its own before touching React:

```bash
curl -X POST -F "image=@some_test_photo.jpg" http://localhost:5000/api/predict
```

Should return something like:
```json
{
  "label": "A",
  "confidence": 0.997,
  "top3": [
    {"label": "A", "confidence": 0.997},
    {"label": "E", "confidence": 0.002},
    {"label": "I", "confidence": 0.001}
  ]
}
```

Or just open http://localhost:5000/api/health in your browser - it
should return `{"status": "ok", "num_classes": 29}`.

## Running both frontend and backend together

You need TWO terminals running at the same time:
- Terminal 1: this backend folder, venv active, `python app.py`
- Terminal 2: your frontend folder, `npm run dev`

The frontend's `API_BASE` already points to `http://localhost:5000`
by default, so no extra config is needed for local development.