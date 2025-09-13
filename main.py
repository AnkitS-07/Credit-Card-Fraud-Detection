import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

def load_model_files():
    required_files = ['fraud_detection_model.pkl', 'scaler.pkl', 'scaler_columns.pkl', 'creditcard.csv']
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"Required file '{file}' not found!")
            st.stop()
    model = joblib.load('fraud_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
    scaler_columns = joblib.load('scaler_columns.pkl')
    full_df = pd.read_csv('creditcard.csv')
    return model, scaler, scaler_columns, full_df

def convert_currency(amount, from_currency, rates_dict):
    if from_currency not in rates_dict:
        st.warning(f"Currency '{from_currency}' not supported. Using EUR.")
        return amount
    return amount / rates_dict[from_currency]

def get_actionable_recommendations():
    return [
        "Contact your bank immediately.",
        "Freeze your card/account temporarily.",
        "Review your recent account activity.",
        "Report the fraud to your bank's fraud department."
    ]

st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")

model, scaler, scaler_columns, full_df = load_model_files()

currency_rates = {
    'EUR': 1.0, 'USD': 1.08, 'GBP': 0.87, 'INR': 90.0, 'JPY': 150.0,
    'AUD': 1.6, 'CAD': 1.45, 'CHF': 0.99, 'CNY': 7.45, 'SGD': 1.46
}

fraud_indices = [6108, 6331, 6334, 6336, 6338, 6427,
                 6446, 6472, 6529, 6609, 6641, 6717, 6719, 6734, 6774, 6820,
                 6870, 6882, 6899, 6903, 6971, 8296, 8312, 8335, 8615, 8617,
                 8842, 8845, 8972, 9035, 9179, 9252, 9487, 9509, 10204, 10484,
                 10497, 10498, 10568, 10630, 10690, 10801, 10891, 10897, 11343,
                 11710, 11841, 11880, 12070, 12108, 12261, 12369, 14104, 14170,
                 14197, 14211, 14338, 15166, 15204, 15225, 15451, 15476,
                 15506, 15539, 15566, 15736, 15751, 15781, 15810, 16415,
                 16780, 16863, 17317, 17366, 17407, 17453, 17480, 18466,
                 18472, 18773, 18809, 20198, 23308, 23422, 26802, 27362,
                 27627, 27738, 27749, 29687, 30100, 30314, 30384, 30398,
                 30442, 30473]

st.subheader("Transaction Input")
time_input = st.number_input("Transaction Time (seconds)", min_value=0.0, value=0.0, step=1.0)
amount_input = st.number_input("Transaction Amount", min_value=0.0, value=0.0, step=0.01)
currency = st.selectbox("Select Currency", list(currency_rates.keys()))
amount_eur = convert_currency(amount_input, currency, currency_rates)
st.write(f"Converted Amount in EUR: {amount_eur:.2f} €")

use_index = st.checkbox("Use specific real transaction index (Demo Purpose)")

transaction_index = None
if use_index:
    st.markdown("### Select Known Fraud Index (Demo Purpose) or Enter Custom Index")
    selected_predefined_index = st.selectbox(
        "Select Fraudulent Transaction Index (Known)", [None] + fraud_indices
    )
    custom_index_input = st.text_input("Or type a custom transaction index (integer)", "")

    if selected_predefined_index is not None:
        transaction_index = selected_predefined_index
    elif custom_index_input.isdigit():
        transaction_index = int(custom_index_input)

if use_index and transaction_index is not None:
    transaction = full_df.loc[transaction_index]
    st.write(f"Showing transaction at index {transaction_index}:")
    st.write(transaction)
    pca_features = transaction[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                                'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                                'V21','V22','V23','V24','V25','V26','V27','V28']].values
else:
    transaction = full_df.sample(1, random_state=np.random.randint(1000)).iloc[0]
    pca_features = transaction[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
                                'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
                                'V21','V22','V23','V24','V25','V26','V27','V28']].values

input_features = np.concatenate(([time_input, amount_eur], pca_features)).reshape(1, -1)
input_features[:, :2] = scaler.transform(input_features[:, :2]) 
fraud_threshold = 0.05

if st.button("Predict Fraud"):
    fraud_prob = model.predict_proba(input_features)[0][1]
    prediction = 1 if fraud_prob > fraud_threshold else 0

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! Probability: {fraud_prob:.2%}")
        st.write("Recommended Actions:")
        for action in get_actionable_recommendations():
            st.write(f"• {action}")
    else:
        st.success(f"✅ Transaction appears Genuine. Fraud Probability: {fraud_prob:.2%}")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:12px;">Made with 💡 by Ankit Sarkar</p>', unsafe_allow_html=True)
