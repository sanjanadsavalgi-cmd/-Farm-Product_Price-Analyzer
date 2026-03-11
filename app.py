# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="Farm Product Price Analyzer", layout="wide")
st.title("🌾 Farm Product Price Analyzer")

# -----------------------------
# 1. Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("farm_price_dataset.csv")
    df.columns = df.columns.str.strip()  # Remove extra spaces
    return df

df = load_data()

# Show all products in the table
st.subheader("Dataset Preview (All Products)")
st.dataframe(df)

# -----------------------------
# 2. User Selection
# -----------------------------
products = df['Product'].unique()
selected_product = st.selectbox("Select Product for Prediction", products)

# Filter dataset for the selected product
data = df[df['Product'] == selected_product]

# -----------------------------
# 3. Prepare Data
# -----------------------------
X = data[['Demand']]  # Using Demand as input
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. Train Model
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
st.subheader("Model Performance")
st.write(f"R² Score: {r2:.2f}")

# -----------------------------
# 5. Predict Price for User Input
# -----------------------------
demand_input = st.number_input(
    f"Enter expected demand for {selected_product}",
    min_value=int(df['Demand'].min()),
    max_value=int(df['Demand'].max()),
    value=int(df['Demand'].min())
)

predicted_price = model.predict(np.array([[demand_input]]))[0]
st.success(f"Predicted Price: ₹{predicted_price:.2f}")

# -----------------------------
# 6. Single Bar Graph: Average Price per Product
# -----------------------------
st.subheader("Average Price per Product")

avg_price = df.groupby('Product')['Price'].mean().reset_index()

fig, ax = plt.subplots()
ax.bar(avg_price['Product'], avg_price['Price'], color='skyblue')
ax.set_xlabel("Product")
ax.set_ylabel("Average Price")
ax.set_title("Average Price of All Products")
st.pyplot(fig)