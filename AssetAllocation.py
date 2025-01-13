import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Step 1: Load the data
file_path = "C:/Users/Tange/Downloads/HeadStarterProject/MABData.csv"  # Update with your file path
data = pd.read_csv(file_path)

# Drop unnecessary columns (e.g., 'Data' as dates) and define X, y
X = data.drop(columns=["Y", "Data"])  # Features
y = data["Y"]  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train logistic regression with class weights
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Step 2: Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)  # Predicted classes
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Predicted probabilities

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC-ROC Score:", auc_score)

# Step 3: Define the investment strategy
def allocate_assets(prediction):
    """
    Asset allocation logic based on the model's prediction.
    - If `Y=1` (Anomaly): Shift to safer assets.
    - If `Y=0` (No Anomaly): Focus on high-return assets.
    """
    if prediction == 1:  # Anomaly predicted
        return {"Equities": 0.0, "Bonds": 0.5, "Gold": 0.3, "Cash": 0.2}
    else:  # No anomaly predicted
        return {"Equities": 0.6, "Bonds": 0.2, "Gold": 0.1, "Cash": 0.1}

# Step 4: Simulate backtesting
# Simulate portfolio returns with historical asset returns and model predictions
historical_data = pd.DataFrame({
    "Date": data["Data"],  # Assuming 'Data' column contains dates
    "Equities_Return": np.random.normal(0.01, 0.02, size=len(data)),  # Simulated returns
    "Bonds_Return": np.random.normal(0.005, 0.01, size=len(data)),
    "Gold_Return": np.random.normal(0.007, 0.015, size=len(data)),
    "Cash_Return": np.full(len(data), 0.002),  # Fixed return for cash
})

# Add model predictions
historical_data["Prediction"] = model.predict(scaler.transform(X))

# Apply allocation logic and calculate portfolio returns
def calculate_portfolio_return(row):
    allocation = allocate_assets(row["Prediction"])
    return (
        row["Equities_Return"] * allocation["Equities"]
        + row["Bonds_Return"] * allocation["Bonds"]
        + row["Gold_Return"] * allocation["Gold"]
        + row["Cash_Return"] * allocation["Cash"]
    )

historical_data["Portfolio_Return"] = historical_data.apply(calculate_portfolio_return, axis=1)

# Calculate cumulative returns
historical_data["Cumulative_Return"] = (1 + historical_data["Portfolio_Return"]).cumprod()

# Step 5: Add benchmark comparison
# Simulate a benchmark portfolio (e.g., 60% equities, 40% bonds)
historical_data["Benchmark_Return"] = (
    0.6 * historical_data["Equities_Return"] + 0.4 * historical_data["Bonds_Return"]
)
historical_data["Benchmark_Cumulative_Return"] = (1 + historical_data["Benchmark_Return"]).cumprod()

# Step 6: Evaluate performance metrics
# Total portfolio return
total_return = historical_data["Cumulative_Return"].iloc[-1] - 1
print("Total Portfolio Return:", total_return)

# Annualized return
num_years = len(historical_data) / 252  # Assuming 252 trading days in a year
annualized_return = (historical_data["Cumulative_Return"].iloc[-1]) ** (1 / num_years) - 1
print("Annualized Return:", annualized_return)

# Maximum drawdown
max_drawdown = (historical_data["Cumulative_Return"] / historical_data["Cumulative_Return"].cummax() - 1).min()
print("Maximum Drawdown:", max_drawdown)

# Step 7: Plot cumulative returns


# Convert the Date column to datetime format
historical_data["Date"] = pd.to_datetime(historical_data["Date"])

# Plot cumulative returns with cleaner date formatting
plt.figure(figsize=(12, 6))
plt.plot(historical_data["Date"], historical_data["Cumulative_Return"], label="Strategy")
plt.plot(historical_data["Date"], historical_data["Benchmark_Cumulative_Return"], label="Benchmark", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.title("Portfolio Cumulative Return vs. Benchmark")
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis for better readability

# Adjust x-axis ticks to show fewer labels
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Show only 10 date labels
plt.grid()
plt.show()
