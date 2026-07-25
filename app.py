import streamlit as st
import pickle
import numpy as np


# Load trained models
churn_model = pickle.load(open("churn_model.pkl", "rb"))

segmentation_model = pickle.load(
    open("segmentation_model.pkl", "rb")
)

segmentation_scaler = pickle.load(
    open("segmentation_scaler.pkl", "rb")
)


# Title
st.title("Customer Segmentation & Churn Analysis")


# -----------------------------
# CHURN PREDICTION
# -----------------------------

st.header("Customer Churn Prediction")

st.write("Enter customer details:")

churn_inputs = []

for i in range(churn_model.n_features_in_):
    value = st.number_input(
        f"Feature {i+1}",
        value=0.0
    )
    churn_inputs.append(value)


if st.button("Predict Churn"):

    prediction = churn_model.predict(
        np.array(churn_inputs).reshape(1, -1)
    )[0]

    if prediction == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")


# -----------------------------
# CUSTOMER SEGMENTATION
# -----------------------------

st.header("Customer Segmentation")

segment_inputs = []

for i in range(segmentation_model.n_features_in_):
    value = st.number_input(
        f"Segmentation Feature {i+1}",
        value=0.0,
        key=f"segment_{i}"
    )
    segment_inputs.append(value)


if st.button("Find Customer Segment"):

    scaled_data = segmentation_scaler.transform(
        np.array(segment_inputs).reshape(1, -1)
    )

    segment = segmentation_model.predict(
        scaled_data
    )[0]

    st.info(
        f"Customer belongs to Segment: {int(segment)}"
    )


   
      

   
