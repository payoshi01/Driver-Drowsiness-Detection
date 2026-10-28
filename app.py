import base64
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from detector import DriverMonitor

app = Flask(__name__)
monitor = DriverMonitor()   # single local user; create per-session for multi-user


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)["image"].split(",", 1)[1]
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "bad frame"}), 400
    return jsonify(monitor.process(frame))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
