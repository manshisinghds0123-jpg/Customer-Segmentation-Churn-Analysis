import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = pd.read_csv("customer_churn.csv")

print("Dataset Loaded Successfully")
print(data.head())


# Convert Total Charges into numeric
data["Total Charges"] = pd.to_numeric(
    data["Total Charges"], 
    errors="coerce"
)

# Fill missing values
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

data.drop(columns=drop_columns, inplace=True)


# Separate features and target
X = data.drop("Churn Value", axis=1)
y = data["Churn Value"]


# Encode categorical columns
encoder = LabelEncoder()

for column in X.select_dtypes(include="object").columns:
    X[column] = encoder.fit_transform(X[column])


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Prediction
prediction = model.predict(X_test)


# Evaluation
accuracy = accuracy_score(
    y_test,
    prediction
)

print("Model Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        prediction
    )
)


# Save model
with open("churn_model.pkl", "wb") as file:
    pickle.dump(model, file)


print("Model Saved Successfully as churn_model.pkl")