from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging

import mlib

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)


@app.route("/")
def home():
    html = f"<h3>Predict whether passenger survived the titanic accident. </h3>"
    return html.format(format)


@app.route("/predict", methods=["POST"])
def predict():
    """Predicts whether passenger survived the titanic accident"""

    json_payload = request.json
    LOG.info(f"JSON payload: {json_payload}")
    prediction = mlib.predict(json_payload["Pclass"],json_payload["Fare"],json_payload["topic_id"],json_payload["Parch"],json_payload["SibSp"],json_payload["retrain"])
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
