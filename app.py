# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('supply_chain_model.pkl')

# Title of the web app
st.title('AI Supply Chain Optimization')

# User input for the day to predict demand
st.subheader('Enter Day for Demand Prediction')
day_input = st.number_input('Enter the Day (e.g., 101)', min_value=1, max_value=200, value=101)

# Predict demand for the input day
predicted_demand = model.predict(np.array([[day_input]]))[0]

# Display the predicted demand
st.write(f"Predicted Demand for Day {day_input}: {predicted_demand:.2f}")

# Display the demand and supply data over time
st.subheader("Demand & Supply Data Visualization")
df = pd.read_csv('synthetic_supply_chain_data.csv')  # Load the data for visualization
st.line_chart(df[['Demand', 'Supply']])

# Display the cumulative demand vs cumulative supply chart
st.subheader("Cumulative Demand vs Cumulative Supply")
st.line_chart(df[['Cumulative_Demand', 'Cumulative_Supply']])

# Optimization Suggestions (simple logic for demonstration)
st.subheader("Optimization Suggestions")
if predicted_demand > df['Supply'].max():
    st.write("Warning: Predicted demand exceeds available supply. Consider increasing supply or finding alternative suppliers.")
else:
    st.write("Supply is sufficient to meet the predicted demand.")

