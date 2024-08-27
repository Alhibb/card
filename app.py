import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model, category mappings, and scaler
model = joblib.load('credit_model.pkl')
category_mappings = joblib.load('category_mappings.pkl')
scaler = joblib.load('scaler.pkl')

# Define function to preprocess input data
def preprocess_data(data):
    categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna('missing').astype(str)
            if col in category_mappings:
                le = LabelEncoder()
                le.classes_ = category_mappings[col]
                data[col] = le.transform(data[col])
            else:
                raise ValueError(f"Unexpected category in column {col}")
    
    numerical_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
    if not all(col in data.columns for col in numerical_cols):
        raise ValueError(f"Data must include columns: {', '.join(numerical_cols)}")
    data[numerical_cols] = scaler.transform(data[numerical_cols])
    
    return data

# Define function to make predictions
def predict_credit_approval(input_data):
    preprocessed_data = preprocess_data(pd.DataFrame([input_data]))
    prediction = model.predict(preprocessed_data)
    return "Approved" if prediction[0] == 1 else "Not Approved"

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .title {font-size: 36px; color: #4B9CD3; text-align: center; font-weight: bold;}
    .expander-header {font-size: 20px; font-weight: bold;}
    .stButton>button {background-color: #4B9CD3; color: white; border-radius: 5px; padding: 10px 20px;}
    .stButton>button:hover {background-color: #3a8cc1;}
    .stAlert {font-size: 18px;}
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.markdown('<p class="title">Credit Approval Prediction</p>', unsafe_allow_html=True)

# Meanings of A1, A2, etc.
meanings = {
    'A1': 'Checking account status',
    'A2': 'Duration in months',
    'A3': 'Credit history',
    'A4': 'Purpose of the credit',
    'A5': 'Credit amount',
    'A6': 'Savings account/bonds',
    'A7': 'Employment status',
    'A8': 'Installment rate as a percentage of disposable income',
    'A9': 'Personal status and sex',
    'A10': 'Other debtors/guarantors',
    'A11': 'Present residence',
    'A12': 'Property',
    'A13': 'Age',
    'A14': 'Other installment plans',
    'A15': 'Housing situation'
}

# Collect user input with organized sections and explanations
with st.expander("Personal Information"):
    input_data = {}
    input_data['A1'] = st.selectbox(f'A1 ({meanings["A1"]})', options=category_mappings['A1'], help=meanings['A1'])
    input_data['A2'] = st.number_input(f'A2 ({meanings["A2"]})', min_value=0.0, help=meanings['A2'])
    input_data['A3'] = st.number_input(f'A3 ({meanings["A3"]})', min_value=0.0, help=meanings['A3'])

with st.expander("Financial Information"):
    input_data['A4'] = st.selectbox(f'A4 ({meanings["A4"]})', options=category_mappings['A4'], help=meanings['A4'])
    input_data['A5'] = st.selectbox(f'A5 ({meanings["A5"]})', options=category_mappings['A5'], help=meanings['A5'])
    input_data['A6'] = st.selectbox(f'A6 ({meanings["A6"]})', options=category_mappings['A6'], help=meanings['A6'])
    input_data['A7'] = st.selectbox(f'A7 ({meanings["A7"]})', options=category_mappings['A7'], help=meanings['A7'])
    input_data['A8'] = st.number_input(f'A8 ({meanings["A8"]})', min_value=0.0, help=meanings['A8'])

with st.expander("Credit History"):
    input_data['A9'] = st.selectbox(f'A9 ({meanings["A9"]})', options=category_mappings['A9'], help=meanings['A9'])
    input_data['A10'] = st.selectbox(f'A10 ({meanings["A10"]})', options=category_mappings['A10'], help=meanings['A10'])
    input_data['A11'] = st.number_input(f'A11 ({meanings["A11"]})', min_value=0.0, help=meanings['A11'])

with st.expander("Other Information"):
    input_data['A12'] = st.selectbox(f'A12 ({meanings["A12"]})', options=category_mappings['A12'], help=meanings['A12'])
    input_data['A13'] = st.selectbox(f'A13 ({meanings["A13"]})', options=category_mappings['A13'], help=meanings['A13'])
    input_data['A14'] = st.number_input(f'A14 ({meanings["A14"]})', min_value=0.0, help=meanings['A14'])
    input_data['A15'] = st.number_input(f'A15 ({meanings["A15"]})', min_value=0.0, help=meanings['A15'])

# Predict and display result with a progress spinner
if st.button('Predict'):
    with st.spinner('Predicting...'):
        try:
            result = predict_credit_approval(input_data)
            icon = "✅" if result == "Approved" else "❌"
            st.success(f"{icon} Credit Approval Status: {result}", icon="ℹ️")
        except Exception as e:
            st.error(f"Error: {e}")
