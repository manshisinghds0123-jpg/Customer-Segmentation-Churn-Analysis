from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)


# Load trained models
churn_model = pickle.load(
    open("churn_model.pkl", "rb")
)

segmentation_model = pickle.load(
    open("segmentation_model.pkl", "rb")
)

segmentation_scaler = pickle.load(
    open("segmentation_scaler.pkl", "rb")
)


@app.route("/")
def home():
    return "Customer Segmentation & Churn Analysis API Running"


# -----------------------------
# CHURN PREDICTION API
# -----------------------------

@app.route("/predict-churn", methods=["POST"])
def predict_churn():

    data = request.json

    # Get values sent by frontend
    input_values = list(data.values())

    # Check feature count
    expected_features = churn_model.n_features_in_

    if len(input_values) != expected_features:
        return jsonify({
            "error": f"Model expects {expected_features} features but received {len(input_values)}"
        }), 400


    prediction = churn_model.predict(
        np.array(input_values).reshape(1, -1)
    )[0]


    if prediction == 1:
        result = "Customer likely to churn"
    else:
        result = "Customer likely to stay"


    return jsonify({
        "prediction": result
    })


# -----------------------------
# CUSTOMER SEGMENTATION API
# -----------------------------

@app.route("/segment-customer", methods=["POST"])
def segment_customer():

    data = request.json

    input_values = list(data.values())


    scaled_data = segmentation_scaler.transform(
        np.array(input_values).reshape(1, -1)
    )


    segment = segmentation_model.predict(
        scaled_data
    )[0]


    return jsonify({
        "customer_segment": int(segment)
    })

@app.route("/features")
def get_features():
    return jsonify({
        "required_features": int(churn_model.n_features_in_)
    })
if __name__ == "__main__":
    app.run(debug=True)