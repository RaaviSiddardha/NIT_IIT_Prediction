import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("new_data1.csv")

data = load_data()

# Application title
st.title("NIT/IIT Admission Cutoff Analysis and Rank Prediction")
st.markdown("""
This application provides insights into NIT/IIT cutoff data and predicts ranks based on user input.
""")

# Sidebar menu
menu = ["Dataset Overview", "Data Visualization", "Rank Prediction"]
choice = st.sidebar.selectbox("Select Menu", menu)

# Dataset Overview
if choice == "Dataset Overview":
    st.header("Dataset Overview")
    st.write(data.head(10))
    st.write("Dataset Dimensions: ", data.shape)
    st.write("Columns: ", data.columns.tolist())
    st.write("Missing Values: ", data.isnull().sum())

# Data Visualization
elif choice == "Data Visualization":
    st.header("Data Visualization")

    st.subheader("Opening vs Closing Ranks")
    institute_type = st.selectbox("Select Institute Type", data["institute_type"].unique())
    category = st.selectbox("Select Category", data["category"].unique())
    filtered_data = data[(data["institute_type"] == institute_type) & (data["category"] == category)]
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_data, x="opening_rank", y="closing_rank", hue="round_no", ax=ax)
    ax.set_title(f"{institute_type} Opening vs Closing Ranks for {category}")
    st.pyplot(fig)

    st.subheader("Category Distribution")
    category_counts = data["category"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
    ax.set_title("Category Distribution")
    st.pyplot(fig)

# Rank Prediction
elif choice == "Rank Prediction":
    st.header("Rank Prediction")

    # User Input
    st.subheader("Input Details")
    institute_type = st.selectbox("Institute Type", data["institute_type"].unique())
    category = st.selectbox("Category", data["category"].unique())
    program_name = st.selectbox("Program Name", data["program_name"].unique())
    round_no = st.number_input("Counseling Round", min_value=1, max_value=6, step=1)

    # Prepare data for prediction
    X = data[["round_no", "category", "program_name", "institute_type"]]
    X = pd.get_dummies(X, drop_first=True)
    y = data["closing_rank"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Prediction
    if st.button("Predict Rank"):
        input_data = pd.DataFrame({
            "round_no": [round_no],
            "category": [category],
            "program_name": [program_name],
            "institute_type": [institute_type]
        })
        input_data = pd.get_dummies(input_data, drop_first=True).reindex(columns=X.columns, fill_value=0)
        prediction = model.predict(input_data)[0]
        st.write(f"Predicted Closing Rank: {int(prediction)}")

# Footer
st.markdown("---")
st.markdown("**Developed with ❤️ for  Predicting the NIT-IIT Colleges**")
