import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(
    page_title="Fraud Guard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {width: 100%; border-radius: 5px; height: 50px; font-weight: bold;}
    .metric-card {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

current_dir = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data():
    path = os.path.join(current_dir, 'data', 'raw', 'cleaned_creditcard.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        # --- تصحيح أسماء الأعمدة (كبير وصغير) ---
        if 'class' in df.columns:
            df = df.rename(columns={'class': 'Class'})
        if 'amount' in df.columns:
            df = df.rename(columns={'amount': 'Amount'})
        return df
    return None


@st.cache_resource
def load_models():
    rf_path = os.path.join(current_dir, '..', 'models', 'rf_fraud_model.joblib')
    iso_path = os.path.join(current_dir, '..', 'models', 'iso_fraud_model.joblib')

    rf_model, iso_model = None, None

    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
    if os.path.exists(iso_path):
        iso_model = joblib.load(iso_path)

    return rf_model, iso_model


df = load_data()
rf_model, iso_model = load_models()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2058/2058768.png", width=100)
st.sidebar.title("👮‍♂️ Control Panel")
st.sidebar.markdown("---")
st.sidebar.info(
    "This system uses a **Hybrid AI Approach**:\n\n1. **Random Forest:** Checks transaction patterns.\n2. **Isolation Forest:** Detects unusual anomalies.")

if df is not None:
    if 'Class' not in df.columns:
        st.error(f"Error: Column 'Class' not found. Available columns: {list(df.columns)}")
    elif 'Amount' not in df.columns:
        st.error(f"Error: Column 'Amount' not found. Available columns: {list(df.columns)}")
    else:
        st.title("🛡️ Financial Fraud Detection System")
        st.markdown("### Real-time Monitoring Dashboard")

        total_txns = len(df)
        fraud_txns = len(df[df['Class'] == 1])
        fraud_pct = (fraud_txns / total_txns) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{total_txns:,}", "All Records")
        col2.metric("Fraud Cases", f"{fraud_txns}", "Detected", delta_color="inverse")
        col3.metric("Fraud Rate", f"{fraud_pct:.3f}%", "Ratio")

        if rf_model and iso_model:
            col4.success("✅ System Fully Active")
        elif rf_model or iso_model:
            col4.warning("⚠️ Partial Protection")
        else:
            col4.error("❌ Models Not Loaded")

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📊 Transaction Distribution")
            fig_pie = px.pie(df, names='Class', title='Normal vs Fraud',
                             color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}, hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("💰 Fraud Amounts Analysis")
            fraud_data = df[df['Class'] == 1]
            fig_hist = px.histogram(fraud_data, x='Amount', nbins=30,
                                    title='Distribution of Stolen Amounts', color_discrete_sequence=['#e74c3c'])
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("---")
        st.header("🕵️ Live Transaction Scanner")
        st.markdown("Enter transaction details below to scan for fraud:")

        input_col1, input_col2, input_col3 = st.columns(3)

        with input_col1:
            amount_val = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
        with input_col2:
            v1_val = st.number_input("V1 Parameter (Try -10 for Fraud)", value=0.0)
        with input_col3:
            v4_val = st.number_input("V4 Parameter (Try 5.0 for Fraud)", value=0.0)

        if st.button("🚨 SCAN TRANSACTION NOW"):
            if rf_model is None or iso_model is None:
                st.error("Please run 'train.py' and 'train_iso.py' first!")
            else:
                input_features = np.zeros((1, 30))

                input_features[0, 29] = amount_val
                input_features[0, 1] = v1_val
                input_features[0, 4] = v4_val

                iso_pred = iso_model.predict(input_features)[0]

                rf_pred = rf_model.predict(input_features)[0]
                rf_prob = rf_model.predict_proba(input_features)[0][1]

                st.subheader("🔍 Scan Results:")

                res_col1, res_col2 = st.columns(2)

                with res_col1:
                    if iso_pred == -1:
                        st.error("⚠️ Anomaly Detected (Isolation Forest)")
                        st.caption("This transaction behavior is unusual.")
                    else:
                        st.success("✅ Behavior Normal (Isolation Forest)")

                with res_col2:
                    if rf_pred == 1:
                        st.error(f"🚨 FRAUD DETECTED (Random Forest)")
                        st.write(f"Confidence Level: **{rf_prob * 100:.2f}%**")
                    else:
                        st.success(f"✅ Transaction Approved")
                        st.write(f"Safety Score: **{(1 - rf_prob) * 100:.2f}%**")

else:
    st.error("Data file not found. Please verify the path in 'src/data/raw/'.")

st.markdown("---")
# هنا تم تعديل الأسماء كما طلبت
st.markdown(
    "<div style='text-align: center; color: grey;'>Built with ❤️ by Waleed Abdullah Abdalrahman Hamza | AI Graduation Project 2026</div>",
    unsafe_allow_html=True)