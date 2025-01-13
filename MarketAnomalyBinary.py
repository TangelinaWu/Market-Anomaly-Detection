import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Provide the path to your CSV file
file_path = "C:/Users/Tange/Downloads/HeadStarterProject/MABData.csv"
data = pd.read_csv(file_path)

#getting rid of not num data aka dates, calls it by its name
X = data.drop(columns = ["Y", "Data"]) #different columns of data features
y = data["Y"] #binary numbers 0,1 if market anomaly or not

#check for missing values
print("Missing Values ;\n", X.isnull().sum().sum())

#training the train test split(80% train, 20% test) # ask gpt what this means
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#use logisitc regression model
model = LogisticRegression(max_iter = 1000, random_state = 42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)  # Predicted classes
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Predicted probabilities for class '1'

# Step 3: Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Calculate AUC-ROC Score
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC-ROC Score:", auc_score)

# Step 4: Analyze feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": np.abs(model.coef_[0])  # Coefficients represent feature importance
})
print("\nFeature Importance:\n", feature_importance.sort_values(by="Importance", ascending=False))




