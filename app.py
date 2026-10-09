from flask import Flask, jsonify, request, send_from_directory
from src.inference import SarcasmPredictor
import os

app = Flask(__name__, static_folder="frontend", static_url_path="")

predictor = None
load_error = None
try:
    predictor = SarcasmPredictor("outputs/lstm_attn_ctx.pt")
except SystemExit as e:
    load_error = str(e)
except Exception as e:
    load_error = str(e)

@app.get("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.get("/api/health")
def health():
    return jsonify({"ready": predictor is not None, "error": load_error})

@app.post("/api/predict")
def predict():
    if predictor is None:
        return jsonify({"error": load_error or "Model is not loaded. Train the model first."}), 503
    data = request.get_json(silent=True) or {}
    comment = str(data.get("comment", "")).strip()
    context = str(data.get("context", "")).strip()
    if not comment:
        return jsonify({"error": "Comment is required."}), 400
    try:
        result = predictor.predict(comment, context, top_k=5)
        result["threshold"] = predictor.threshold
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Sarcasm Detector frontend: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
