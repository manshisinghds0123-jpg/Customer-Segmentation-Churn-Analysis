import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans


# Load dataset
data = pd.read_csv("customer_churn.csv")

print("Dataset Loaded Successfully")


# Convert Total Charges
data["Total Charges"] = pd.to_numeric(
    data["Total Charges"],
    errors="coerce"
)

data["Total Charges"] = data["Total Charges"].fillna(
    data["Total Charges"].median()
)


# Remove unnecessary columns
drop_columns = [
    "CustomerID",
    "Country",
    "State",
    "City",
    "Lat Long",
    "Churn Label",
    "Churn Reason"
]

data.drop(
    columns=drop_columns,
    inplace=True
)


# Remove churn-related columns for segmentation
data.drop(
    columns=[
        "Churn Value",
        "Churn Score"
    ],
    inplace=True,
    errors="ignore"
)


# Encode text columns
encoder = LabelEncoder()

for column in data.select_dtypes(include="object").columns:
    data[column] = encoder.fit_transform(data[column])


# Scale data
scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)


# K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)


kmeans.fit(scaled_data)


# Save files
with open("segmentation_model.pkl", "wb") as file:
    pickle.dump(kmeans, file)


with open("segmentation_scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


print("Segmentation Model Saved Successfully")