# model_training.py
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import pandas as pd

# Load the synthetic supply chain data
df = pd.read_csv('synthetic_supply_chain_data.csv')

# Features and target variable
X = df[['Day']]  # Feature: Day
y = df['Demand']  # Target: Demand

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Save the trained model for later use in the Streamlit app
joblib.dump(model, 'supply_chain_model.pkl')

