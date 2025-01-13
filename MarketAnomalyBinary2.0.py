# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Install using `pip install imbalanced-learn`

# Step 1: Load the data
file_path = "C:/Users/Tange/Downloads/HeadStarterProject/MABData.csv"  # Update this with your actual file path
data = pd.read_csv(file_path)

# Drop unnecessary columns (e.g., 'Data' as it's not useful for predictions)
X = data.drop(columns=["Y", "Data"])  # Features
y = data["Y"]  # Target variable

# Step 2: Check and preprocess the data
print("Missing values:\n", X.isnull().sum().sum())  # Check for missing values

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Step 6: Train logistic regression with class weights
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Step 7: Make predictions
y_pred = model.predict(X_test_scaled)  # Predicted classes
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Predicted probabilities

# Step 8: Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Calculate AUC-ROC score
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC-ROC Score:", auc_score)

# Step 9: Analyze feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": np.abs(model.coef_[0])  # Coefficients represent feature importance
})
print("\nFeature Importance:\n", feature_importance.sort_values(by="Importance", ascending=False))

# Step 10: Test alternative thresholds (optional)
threshold = 0.3  # Lower the threshold for classifying anomalies
y_pred_new = (y_pred_proba > threshold).astype(int)
print("\nClassification Report with New Threshold:\n", classification_report(y_test, y_pred_new))
print("Confusion Matrix with New Threshold:\n", confusion_matrix(y_test, y_pred_new))
