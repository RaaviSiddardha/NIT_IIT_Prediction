# NIT & IIT Rank Prediction

## Introduction  
This project is a **Streamlit web application** designed to analyze and predict NIT/IIT admission cut-off ranks based on historical data. The application provides insights into trends and allows users to estimate their expected ranks based on input parameters.

## Tech Stack  
**Python | Machine Learning | Streamlit | Power BI | Render | GitHub**  

## Dataset  
- The dataset consists of past NIT/IIT admission cut-off ranks.  
- It includes factors such as:
  - Exam Scores (JEE Main, JEE Advanced)
  - Category (General, OBC, SC/ST)
  - Branch & College Preferences
  - Past Year Cutoff Trends  
- Data preprocessing includes handling missing values and encoding categorical variables.

## Project Overview  
This end-to-end project includes the following steps:

### 1. Data Collection  
- Aggregation of past NIT/IIT cutoff ranks and student admission data.

### 2. Data Preprocessing  
- Handling missing values, encoding categorical variables, and feature selection.

### 3. Data Visualization  
- Designed an **interactive Streamlit-based user interface** for intuitive data exploration.
- Visualizing trends in **Opening vs. Closing Ranks**.
- Displaying **category distribution** using bar plots.
- Created **dynamic Power BI dashboards** for real-time monitoring of admission predictions.

### 4. Machine Learning Model  
- Developed a **cloud-based predictive model** using **Random Forest Regressor** in Python.
- Achieved a **Root Mean Squared Error (RMSE) of 7,620** on testing data.
- Splitting the dataset into training and testing sets.
- Model evaluation using **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)**.

### 5. Deployment  
- **Deployed the application on Render with GitHub** for seamless accessibility.
- Ensured **cross-functional team collaboration** by making predictions available in real-time.

## Directory Structure  
```
NIT_IIT_Prediction/
│
├── data/                      # Dataset files
│
├── models/                    # Trained ML models
│
├── notebooks/                 # Jupyter notebooks for EDA and model training
│
├── app/                       # Streamlit web application
│   ├── prediction_app.py       # Main application script
│
├── requirements.txt           # List of required dependencies
├── README.md                  # Project documentation
└── main.py                    # Main script to run predictions
```

## Setup Instructions  

### Step 1: Clone the Repository  
```bash
git clone https://github.com/RaaviSiddardha/NIT_IIT_Prediction.git
cd NIT_IIT_Prediction
```

### Step 2: Create a Virtual Environment  
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies  
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application  
```bash
streamlit run app/prediction_app.py
```
The app will be hosted at **http://localhost:8501/** where users can explore admission trends and predict their expected closing ranks.

## Features of the Web App  
✔ **Dataset Overview**: Displaying dataset summary, missing values, and key statistics.  
✔ **Data Visualization**: Graphical insights on opening vs. closing ranks across different categories.  
✔ **Rank Prediction**: Users can input their details and get predicted closing ranks.  
✔ **Power BI Dashboards**: Real-time monitoring of admission predictions for better insights.
https://nit-iit-prediction.onrender.com

## Model Architecture  
The project uses a **Random Forest Regressor** to estimate admission cut-off ranks.  
- Handles non-linear patterns in data.  
- Provides robust predictions based on multiple input features.  

## Future Enhancements  
🚀 **Deploy on Cloud**: Host the application on **Heroku** or **AWS**.  
📊 **Enhance Model Accuracy**: Experiment with **Gradient Boosting** for improved predictions.  
🌐 **Real-Time Data Updates**: Integrate live admission cut-off data for real-time predictions.  



