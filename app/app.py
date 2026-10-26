import streamlit as st
import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
from datetime import datetime
import random
import string


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "fraudshield_xgb_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    import joblib

    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def create_shap_explainer():

    xgb_model = model.named_steps["model"]

    return shap.TreeExplainer(xgb_model)


shap_explainer = create_shap_explainer()


# ============================================================
# MODEL FEATURE LISTS
# ============================================================

numerical_columns = [
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "day_index",
    "hour",
    "minute",
    "has_identity"
]

categorical_columns = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "device_family"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_level(score):

    if score < 30:
        return "LOW RISK"

    elif score < 60:
        return "MEDIUM RISK"

    else:
        return "HIGH RISK"


def get_recommendation(risk_level):

    if risk_level == "LOW RISK":
        return "NORMAL PROCESSING"

    elif risk_level == "MEDIUM RISK":
        return "REVIEW TRANSACTION"

    return "INVESTIGATE TRANSACTION"


def generate_transaction_id():

    random_part = "".join(
        random.choices(
            string.ascii_uppercase + string.digits,
            k=6
        )
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d"
    )

    return f"FS-{timestamp}-{random_part}"


def get_status_message(risk_level):

    if risk_level == "LOW RISK":

        return (
            "🟢 TRANSACTION APPEARS SAFE",
            "The transaction shows no major suspicious signals."
        )

    elif risk_level == "MEDIUM RISK":

        return (
            "🟡 TRANSACTION REQUIRES REVIEW",
            "Some unusual signals were detected and additional verification may be appropriate."
        )

    else:

        return (
            "🔴 TRANSACTION REQUIRES INVESTIGATION",
            "Multiple strong risk signals were detected."
        )


def clean_feature_name(feature):

    feature = feature.replace(
        "num__",
        ""
    )

    feature = feature.replace(
        "cat__",
        ""
    )

    if feature.startswith("ProductCD_"):

        return (
            "Product Category — "
            + feature.replace(
                "ProductCD_",
                ""
            )
        )

    if feature.startswith("card4_"):

        return (
            "Card Network — "
            + feature.replace(
                "card4_",
                ""
            ).title()
        )

    if feature.startswith("card6_"):

        return (
            "Card Type — "
            + feature.replace(
                "card6_",
                ""
            ).title()
        )

    if feature.startswith("P_emaildomain_"):

        return (
            "Purchaser Email — "
            + feature.replace(
                "P_emaildomain_",
                ""
            )
        )

    if feature.startswith("R_emaildomain_"):

        return (
            "Recipient Email — "
            + feature.replace(
                "R_emaildomain_",
                ""
            )
        )

    if feature.startswith("DeviceType_"):

        return (
            "Device Type — "
            + feature.replace(
                "DeviceType_",
                ""
            ).title()
        )

    if feature.startswith("device_family_"):

        return (
            "Device Family — "
            + feature.replace(
                "device_family_",
                ""
            ).title()
        )

    friendly_names = {

        "TransactionDT":
            "Transaction Time Signal",

        "TransactionAmt":
            "Transaction Amount",

        "card1":
            "Card Signal 1",

        "card2":
            "Card Signal 2",

        "card3":
            "Card Signal 3",

        "card5":
            "Card Signal 5",

        "addr1":
            "Address Signal 1",

        "addr2":
            "Address Signal 2",

        "dist1":
            "Distance Signal 1",

        "dist2":
            "Distance Signal 2",

        "day_index":
            "Day Signal",

        "hour":
            "Transaction Hour",

        "minute":
            "Transaction Minute",

        "has_identity":
            "Identity Availability"
    }

    return friendly_names.get(
        feature,
        feature
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def generate_shap_explanation(transaction_df):

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    transformed_transaction = (
        preprocessor.transform(
            transaction_df
        )
    )

    shap_values = (
        shap_explainer.shap_values(
            transformed_transaction
        )
    )

    if isinstance(
        shap_values,
        list
    ):

        shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:

        shap_values = (
            shap_values[:, :, 1]
        )

    shap_values = shap_values[0]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    shap_df = pd.DataFrame({

        "Feature":
            feature_names,

        "SHAP Value":
            shap_values
    })

    shap_df["Importance"] = (
        shap_df["SHAP Value"].abs()
    )

    shap_df = shap_df.sort_values(
        "Importance",
        ascending=False
    )

    shap_df["Feature"] = (
        shap_df["Feature"].apply(
            clean_feature_name
        )
    )

    return shap_df


# ============================================================
# SIMULATOR RISK ENGINE
# ============================================================

def calculate_simulator_risk(
    amount,
    payment_method,
    merchant_category,
    transaction_type,
    location,
    region,
    device,
    new_device,
    new_merchant,
    recent_transactions,
    distance,
    transaction_hour
):

    risk_score = 8.0

    risk_factors = []

    safe_factors = []

    # --------------------------------------------------------
    # AMOUNT
    # --------------------------------------------------------

    if amount >= 100000:

        risk_score += 32

        risk_factors.append(
            "Very high transaction amount"
        )

    elif amount >= 50000:

        risk_score += 24

        risk_factors.append(
            "High transaction amount"
        )

    elif amount >= 20000:

        risk_score += 12

        risk_factors.append(
            "Above-normal transaction amount"
        )

    elif amount <= 5000:

        safe_factors.append(
            "Transaction amount is within a normal range"
        )

    else:

        risk_score += 3

        safe_factors.append(
            "Transaction amount is moderate"
        )

    # --------------------------------------------------------
    # PAYMENT METHOD
    # --------------------------------------------------------

    if payment_method == "Bank Transfer":

        risk_score += 4

    elif payment_method == "UPI":

        risk_score += 2

    else:

        risk_score += 1

    # --------------------------------------------------------
    # TRANSACTION TYPE
    # --------------------------------------------------------

    if transaction_type == "Transfer":

        risk_score += 6

        risk_factors.append(
            "Funds transfer transaction"
        )

    elif transaction_type == "Withdrawal":

        risk_score += 8

        risk_factors.append(
            "Cash withdrawal transaction"
        )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if new_device == "Yes":

        risk_score += 20

        risk_factors.append(
            "New device detected"
        )

    else:

        safe_factors.append(
            "Known device"
        )

    # --------------------------------------------------------
    # MERCHANT
    # --------------------------------------------------------

    if new_merchant == "Yes":

        risk_score += 15

        risk_factors.append(
            "New merchant detected"
        )

    else:

        safe_factors.append(
            "Known merchant"
        )

    # --------------------------------------------------------
    # REGION
    # --------------------------------------------------------

    if region == "International":

        risk_score += 18

        risk_factors.append(
            "International transaction"
        )

    else:

        safe_factors.append(
            "Domestic transaction"
        )

    # --------------------------------------------------------
    # FREQUENCY
    # --------------------------------------------------------

    if recent_transactions >= 15:

        risk_score += 25

        risk_factors.append(
            "Very high transaction frequency"
        )

    elif recent_transactions >= 8:

        risk_score += 18

        risk_factors.append(
            "High transaction frequency"
        )

    elif recent_transactions >= 5:

        risk_score += 10

        risk_factors.append(
            "Multiple recent transactions"
        )

    else:

        safe_factors.append(
            "Normal recent transaction frequency"
        )

    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    if distance >= 500:

        risk_score += 25

        risk_factors.append(
            "Transaction is extremely far from usual location"
        )

    elif distance >= 100:

        risk_score += 18

        risk_factors.append(
            "Transaction is far from usual location"
        )

    elif distance >= 50:

        risk_score += 10

        risk_factors.append(
            "Transaction is unusually far from usual location"
        )

    else:

        safe_factors.append(
            "Transaction is close to usual location"
        )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    if 0 <= transaction_hour <= 4:

        risk_score += 14

        risk_factors.append(
            "Very unusual transaction time"
        )

    elif transaction_hour >= 23:

        risk_score += 8

        risk_factors.append(
            "Late-night transaction"
        )

    elif 5 <= transaction_hour <= 7:

        risk_score += 4

    else:

        safe_factors.append(
            "Transaction occurs during a typical time window"
        )

    # --------------------------------------------------------
    # DEVICE TYPE
    # --------------------------------------------------------

    if device == "Tablet":

        risk_score += 2

    elif device == "Desktop":

        risk_score += 1

    # --------------------------------------------------------
    # MERCHANT CATEGORY
    # --------------------------------------------------------

    if merchant_category in [
        "Shopping",
        "Travel"
    ]:

        risk_score += 2

    elif merchant_category == "Other":

        risk_score += 3

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    if location == "Other":

        risk_score += 2

    # --------------------------------------------------------
    # INTERACTION SIGNALS
    # --------------------------------------------------------

    if (
        new_device == "Yes"
        and new_merchant == "Yes"
    ):

        risk_score += 8

        risk_factors.append(
            "New device and new merchant combination"
        )

    if (
        region == "International"
        and distance >= 100
    ):

        risk_score += 8

        risk_factors.append(
            "International transaction with unusual location"
        )

    if (
        recent_transactions >= 8
        and amount >= 20000
    ):

        risk_score += 8

        risk_factors.append(
            "High transaction frequency combined with high amount"
        )

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    risk_score = min(
        max(
            risk_score,
            1
        ),
        99
    )

    return (
        round(
            risk_score,
            2
        ),
        risk_factors,
        safe_factors
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🛡️ FraudShield"
)

st.sidebar.caption(
    "Transaction Fraud Detection & Risk Analysis"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "💳 Transaction Simulator",
        "🤖 About Model"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "FraudShield is an ML-based fraud-risk assessment "
    "prototype built using the IEEE-CIS Fraud Detection dataset."
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "🛡️ FraudShield"
    )

    st.subheader(
        "Transaction Fraud Detection & Risk Analysis System"
    )

    st.write(
        "Analyze transaction signals using a trained "
        "XGBoost machine learning model."
    )

    st.info(
        "🔬 Research Mode — This page uses the technical "
        "features expected by the IEEE-CIS trained model."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Model",
            "XGBoost"
        )

    with col2:

        st.metric(
            "ROC-AUC",
            "90.3%"
        )

    with col3:

        st.metric(
            "PR-AUC",
            "58.1%"
        )

    with col4:

        st.metric(
            "Input Features",
            "21"
        )

    st.divider()

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    st.header(
        "🔍 Transaction Analysis"
    )

    st.write(
        "Enter the transaction signals used by the "
        "FraudShield research model."
    )

    st.subheader(
        "💰 Transaction Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        TransactionDT = st.number_input(
            "Transaction Time Signal",
            value=100000,
            min_value=0
        )

        TransactionAmt = st.number_input(
            "Transaction Amount",
            value=100.0,
            min_value=0.0
        )

        card1 = st.number_input(
            "Card Signal 1",
            value=1000,
            min_value=0
        )

        card2 = st.number_input(
            "Card Signal 2",
            value=100,
            min_value=0
        )

    with col2:

        card3 = st.number_input(
            "Card Signal 3",
            value=150,
            min_value=0
        )

        card5 = st.number_input(
            "Card Signal 5",
            value=100,
            min_value=0
        )

        addr1 = st.number_input(
            "Address Signal 1",
            value=200,
            min_value=0
        )

        addr2 = st.number_input(
            "Address Signal 2",
            value=80,
            min_value=0
        )

    with col3:

        dist1 = st.number_input(
            "Distance Signal 1",
            value=10.0,
            min_value=0.0
        )

        dist2 = st.number_input(
            "Distance Signal 2",
            value=10.0,
            min_value=0.0
        )

        day_index = st.number_input(
            "Day Signal",
            value=1,
            min_value=0
        )

        hour = st.slider(
            "Transaction Hour",
            0,
            23,
            12
        )

    minute = st.slider(
        "Transaction Minute",
        0,
        59,
        0
    )

    st.subheader(
        "📱 Device & Identity"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        has_identity = st.selectbox(
            "Identity Information Available",
            [1, 0],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        DeviceType = st.selectbox(
            "Device Type",
            [
                "unknown",
                "desktop",
                "mobile"
            ]
        )

    with col2:

        device_family = st.selectbox(
            "Device Family",
            [
                "unknown",
                "windows",
                "ios",
                "mac",
                "android",
                "samsung",
                "trident",
                "chrome",
                "firefox",
                "other"
            ]
        )

        ProductCD = st.selectbox(
            "Product Category",
            [
                "W",
                "H",
                "C",
                "S",
                "R"
            ]
        )

    with col3:

        card4 = st.selectbox(
            "Card Network",
            [
                "visa",
                "mastercard",
                "discover",
                "american express"
            ]
        )

        card6 = st.selectbox(
            "Card Type",
            [
                "credit",
                "debit"
            ]
        )

    st.subheader(
        "📧 Email Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        P_emaildomain = st.selectbox(
            "Purchaser Email Domain",
            [
                "gmail",
                "yahoo",
                "hotmail",
                "outlook",
                "icloud",
                "anonymous",
                "unknown"
            ]
        )

    with col2:

        R_emaildomain = st.selectbox(
            "Recipient Email Domain",
            [
                "gmail",
                "yahoo",
                "hotmail",
                "outlook",
                "icloud",
                "anonymous",
                "unknown"
            ]
        )

    st.divider()

    st.caption(
        "⚙️ Advanced system signals are normally collected "
        "automatically by payment platforms."
    )

    predict_button = st.button(
        "🔍 Analyze Transaction",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        transaction_data = {

            "TransactionDT":
                TransactionDT,

            "TransactionAmt":
                TransactionAmt,

            "card1":
                card1,

            "card2":
                card2,

            "card3":
                card3,

            "card5":
                card5,

            "addr1":
                addr1,

            "addr2":
                addr2,

            "dist1":
                dist1,

            "dist2":
                dist2,

            "day_index":
                day_index,

            "hour":
                hour,

            "minute":
                minute,

            "has_identity":
                has_identity,

            "ProductCD":
                ProductCD,

            "card4":
                card4,

            "card6":
                card6,

            "P_emaildomain":
                P_emaildomain,

            "R_emaildomain":
                R_emaildomain,

            "DeviceType":
                DeviceType,

            "device_family":
                device_family
        }

        transaction_df = pd.DataFrame([
            transaction_data
        ])

        fraud_probability = float(
            model.predict_proba(
                transaction_df
            )[0, 1]
        )

        risk_score = round(
            fraud_probability * 100,
            2
        )

        risk_level = get_risk_level(
            risk_score
        )

        recommendation = get_recommendation(
            risk_level
        )

        st.divider()

        st.header(
            "🛡️ FraudShield Analysis"
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Fraud Probability",
                f"{fraud_probability * 100:.2f}%"
            )

        with result_col2:

            st.metric(
                "Risk Score",
                f"{risk_score:.2f}/100"
            )

        with result_col3:

            st.metric(
                "Risk Level",
                risk_level
            )

        st.progress(
            float(fraud_probability)
        )

        if risk_level == "LOW RISK":

            st.success(
                f"🟢 {recommendation}"
            )

        elif risk_level == "MEDIUM RISK":

            st.warning(
                f"🟡 {recommendation}"
            )

        else:

            st.error(
                f"🔴 {recommendation}"
            )

        st.divider()

        st.header(
            "🔎 Why did the model make this prediction?"
        )

        shap_df = generate_shap_explanation(
            transaction_df
        )

        positive_factors = shap_df[
            shap_df["SHAP Value"] > 0
        ].head(5)

        negative_factors = shap_df[
            shap_df["SHAP Value"] < 0
        ].head(5)

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "🔴 Increased Fraud Risk"
            )

            if len(
                positive_factors
            ) == 0:

                st.write(
                    "No strong positive risk factors identified."
                )

            else:

                for _, row in positive_factors.iterrows():

                    st.write(
                        f"• {row['Feature']}"
                    )

        with col2:

            st.subheader(
                "🟢 Reduced Fraud Risk"
            )

            if len(
                negative_factors
            ) == 0:

                st.write(
                    "No strong protective factors identified."
                )

            else:

                for _, row in negative_factors.iterrows():

                    st.write(
                        f"• {row['Feature']}"
                    )

        st.subheader(
            "Top Model Signals"
        )

        chart_data = (
            shap_df
            .head(10)
            .sort_values(
                "SHAP Value"
            )
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.barh(
            chart_data["Feature"],
            chart_data["SHAP Value"]
        )

        ax.set_xlabel(
            "SHAP Value"
        )

        ax.set_ylabel(
            "Feature"
        )

        ax.set_title(
            "Top Factors Influencing This Prediction"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# TRANSACTION SIMULATOR
# ============================================================

elif page == "💳 Transaction Simulator":

    st.title(
        "💳 Transaction Simulator"
    )

    st.write(
        "Simulate an everyday digital payment and "
        "see how FraudShield assesses its risk."
    )

    st.warning(
        "🛡️ DEMO MODE — No real banking transaction "
        "is performed, processed, or sent to any bank."
    )

    st.divider()

    # --------------------------------------------------------
    # PAYMENT DETAILS
    # --------------------------------------------------------

    st.header(
        "💰 Payment Details"
    )

    col1, col2 = st.columns(2)

    with col1:

        sim_amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=1.0,
            value=2500.0,
            step=100.0
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "UPI",
                "Debit Card",
                "Credit Card",
                "Bank Transfer"
            ]
        )

        merchant_category = st.selectbox(
            "Merchant Category",
            [
                "Shopping",
                "Food & Dining",
                "Travel",
                "Bills & Utilities",
                "Entertainment",
                "Healthcare",
                "Education",
                "Other"
            ]
        )

    with col2:

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "Purchase",
                "Transfer",
                "Bill Payment",
                "Subscription",
                "Withdrawal"
            ]
        )

        transaction_location = st.selectbox(
            "Transaction Location",
            [
                "Delhi",
                "Mumbai",
                "Bengaluru",
                "Hyderabad",
                "Chennai",
                "Kolkata",
                "Other"
            ]
        )

        region_type = st.selectbox(
            "Transaction Region",
            [
                "Domestic",
                "International"
            ]
        )

    st.divider()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    st.header(
        "📱 Device & Behaviour"
    )

    col1, col2 = st.columns(2)

    with col1:

        simulator_device = st.selectbox(
            "Device",
            [
                "Mobile",
                "Desktop",
                "Tablet"
            ]
        )

        new_device = st.selectbox(
            "New Device?",
            [
                "No",
                "Yes"
            ]
        )

        new_merchant = st.selectbox(
            "New Merchant?",
            [
                "No",
                "Yes"
            ]
        )

    with col2:

        recent_transactions = st.number_input(
            "Transactions in Last 30 Minutes",
            min_value=0,
            max_value=50,
            value=2
        )

        usual_location_distance = st.number_input(
            "Distance From Usual Location (km)",
            min_value=0.0,
            value=2.0,
            step=1.0
        )

        transaction_hour = st.slider(
            "Transaction Time",
            0,
            23,
            20
        )

    st.caption(
        "These signals represent information that a real "
        "payment platform could collect automatically."
    )

    st.divider()

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    st.header(
        "🛡️ Transaction Analysis"
    )

    analyze_button = st.button(
        "🔍 Analyze Transaction",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        transaction_id = (
            generate_transaction_id()
        )

        analysis_time = datetime.now()

        with st.spinner(
            "Analyzing transaction signals..."
        ):

            (
                simulator_score,
                risk_factors,
                safe_factors
            ) = calculate_simulator_risk(

                amount=sim_amount,

                payment_method=payment_method,

                merchant_category=merchant_category,

                transaction_type=transaction_type,

                location=transaction_location,

                region=region_type,

                device=simulator_device,

                new_device=new_device,

                new_merchant=new_merchant,

                recent_transactions=recent_transactions,

                distance=usual_location_distance,

                transaction_hour=transaction_hour
            )

        risk_level = get_risk_level(
            simulator_score
        )

        recommendation = get_recommendation(
            risk_level
        )

        (
            status_title,
            status_description
        ) = get_status_message(
            risk_level
        )

        st.divider()

        # ----------------------------------------------------
        # TRANSACTION HEADER
        # ----------------------------------------------------

        st.subheader(
            "🧾 Transaction Analysis"
        )

        info1, info2, info3 = st.columns(3)

        with info1:

            st.caption(
                "Transaction ID"
            )

            st.code(
                transaction_id
            )

        with info2:

            st.caption(
                "Analyzed"
            )

            st.write(
                analysis_time.strftime(
                    "%d %b %Y • %I:%M %p"
                )
            )

        with info3:

            st.caption(
                "Payment Method"
            )

            st.write(
                payment_method
            )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        st.divider()

        st.header(
            status_title
        )

        st.write(
            status_description
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score_col1, score_col2, score_col3 = st.columns(3)

        with score_col1:

            st.metric(
                "Risk Score",
                f"{simulator_score:.2f}/100"
            )

        with score_col2:

            st.metric(
                "Risk Level",
                risk_level
            )

        with score_col3:

            st.metric(
                "Decision",
                recommendation
            )

        # ----------------------------------------------------
        # RISK METER
        # ----------------------------------------------------

        st.subheader(
            "📊 Risk Meter"
        )

        st.progress(
            int(simulator_score)
        )

        meter_col1, meter_col2, meter_col3 = st.columns(3)

        with meter_col1:

            st.caption(
                "🟢 LOW\n\n0 – 29"
            )

        with meter_col2:

            st.caption(
                "🟡 MEDIUM\n\n30 – 59"
            )

        with meter_col3:

            st.caption(
                "🔴 HIGH\n\n60 – 100"
            )

        if risk_level == "LOW RISK":

            st.success(
                f"🟢 {recommendation}"
            )

        elif risk_level == "MEDIUM RISK":

            st.warning(
                f"🟡 {recommendation}"
            )

        else:

            st.error(
                f"🔴 {recommendation}"
            )

        # ----------------------------------------------------
        # SECURITY CHECKS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🔐 Security Checks"
        )

        check1, check2 = st.columns(2)

        with check1:

            if new_device == "Yes":

                st.warning(
                    "⚠️ Device verification — New device"
                )

            else:

                st.success(
                    "✅ Device verification — Known device"
                )

            if usual_location_distance > 50:

                st.warning(
                    "⚠️ Location consistency — Unusual distance"
                )

            else:

                st.success(
                    "✅ Location consistency — Normal"
                )

            if recent_transactions >= 8:

                st.warning(
                    "⚠️ Transaction frequency — Elevated"
                )

            else:

                st.success(
                    "✅ Transaction frequency — Normal"
                )

        with check2:

            if sim_amount >= 20000:

                st.warning(
                    "⚠️ Amount analysis — Elevated amount"
                )

            else:

                st.success(
                    "✅ Amount analysis — Normal range"
                )

            if new_merchant == "Yes":

                st.warning(
                    "⚠️ Merchant analysis — New merchant"
                )

            else:

                st.success(
                    "✅ Merchant analysis — Known merchant"
                )

            if region_type == "International":

                st.warning(
                    "⚠️ Region analysis — International"
                )

            else:

                st.success(
                    "✅ Region analysis — Domestic"
                )

        # ----------------------------------------------------
        # RISK INTERPRETATION
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📖 Risk Interpretation"
        )

        if simulator_score < 15:

            st.write(
                "The transaction shows very few suspicious "
                "signals and appears consistent with normal "
                "activity."
            )

        elif simulator_score < 30:

            st.write(
                "The transaction appears relatively normal, "
                "although some minor risk signals may be present."
            )

        elif simulator_score < 60:

            st.write(
                "The transaction contains several signals "
                "that may justify additional verification."
            )

        else:

            st.write(
                "The transaction contains multiple strong "
                "risk signals and should receive additional "
                "investigation."
            )

        # ----------------------------------------------------
        # RISK FACTORS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🔎 Risk Factors"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 🔴 Increased Risk"
            )

            if risk_factors:

                for factor in risk_factors:

                    st.write(
                        f"• {factor}"
                    )

            else:

                st.write(
                    "No major risk signals detected."
                )

        with col2:

            st.markdown(
                "### 🟢 Protective Signals"
            )

            if safe_factors:

                for factor in safe_factors:

                    st.write(
                        f"• {factor}"
                    )

            else:

                st.write(
                    "No strong protective signals detected."
                )

        # ----------------------------------------------------
        # TRANSACTION SUMMARY
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📋 Transaction Summary"
        )

        summary_data = {

            "Transaction ID":
                transaction_id,

            "Amount":
                f"₹{sim_amount:,.2f}",

            "Payment Method":
                payment_method,

            "Merchant Category":
                merchant_category,

            "Transaction Type":
                transaction_type,

            "Location":
                transaction_location,

            "Region":
                region_type,

            "Device":
                simulator_device,

            "New Device":
                new_device,

            "New Merchant":
                new_merchant,

            "Recent Transactions":
                recent_transactions,

            "Distance From Usual Location":
                f"{usual_location_distance:.1f} km",

            "Transaction Time":
                f"{transaction_hour:02d}:00",

            "Risk Score":
                f"{simulator_score:.2f}/100",

            "Risk Level":
                risk_level,

            "Decision":
                recommendation
        }

        summary_df = pd.DataFrame(
            summary_data.items(),
            columns=[
                "Field",
                "Value"
            ]
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # IMPORTANT DISCLOSURE
        # ----------------------------------------------------

        st.divider()

        st.info(
            "ℹ️ Simulator Note: This page uses a transparent "
            "rule-based risk simulation to demonstrate how "
            "a real-world transaction-risk interface could "
            "work. Its score is NOT an XGBoost prediction "
            "and is NOT connected to a bank or payment network."
        )

    st.divider()

    st.caption(
        "Demo / Research Application — FraudShield does not "
        "process real payments or connect to real financial "
        "accounts."
    )


# ============================================================
# ABOUT MODEL
# ============================================================

elif page == "🤖 About Model":

    st.title(
        "🤖 About FraudShield"
    )

    st.write(
        "An end-to-end machine learning project for "
        "transaction fraud detection and risk analysis."
    )

    st.divider()

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    st.header(
        "🎯 Project Objective"
    )

    st.write(
        "FraudShield uses machine learning to identify "
        "transactions that may contain fraudulent behavior."
    )

    st.write(
        "The research model was trained using the "
        "IEEE-CIS Fraud Detection dataset."
    )

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    st.header(
        "📊 Model Performance"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Model",
            "XGBoost"
        )

    with col2:

        st.metric(
            "ROC-AUC",
            "90.3%"
        )

    with col3:

        st.metric(
            "PR-AUC",
            "58.1%"
        )

    with col4:

        st.metric(
            "Training Rows",
            "50,000"
        )

    st.caption(
        "Metrics shown here come from the project's "
        "50,000-row IEEE-CIS development subset and "
        "single 80/20 stratified evaluation split."
    )

    st.divider()

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.header(
        "🔄 Machine Learning Pipeline"
    )

    st.write(
        """
        Data Understanding
        ↓
        Exploratory Data Analysis
        ↓
        Feature Engineering
        ↓
        Data Preprocessing
        ↓
        Logistic Regression Baseline
        ↓
        XGBoost
        ↓
        Class Imbalance Analysis
        ↓
        Threshold Analysis
        ↓
        SHAP Explainability
        ↓
        Risk Assessment
        """
    )

    st.divider()

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.header(
        "🧩 Feature Groups"
    )

    feature_data = pd.DataFrame({

        "Feature Group": [

            "Transaction",

            "Card",

            "Address",

            "Distance",

            "Time",

            "Email",

            "Device",

            "Identity"
        ],

        "Examples": [

            "TransactionAmt, TransactionDT",

            "card1, card2, card3, card5",

            "addr1, addr2",

            "dist1, dist2",

            "day_index, hour, minute",

            "P_emaildomain, R_emaildomain",

            "DeviceType, device_family",

            "has_identity"
        ]
    })

    st.dataframe(
        feature_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    st.header(
        "🔎 Explainable AI with SHAP"
    )

    st.write(
        "FraudShield uses SHAP to understand which model "
        "features contribute most strongly to individual "
        "predictions."
    )

    shap_image_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "images",
        "shap_summary.png"
    )

    if os.path.exists(
        shap_image_path
    ):

        st.image(
            shap_image_path,
            caption="SHAP Feature Importance",
            use_container_width=True
        )

    else:

        st.info(
            "SHAP summary image is not available."
        )

    st.divider()

    # --------------------------------------------------------
    # RISK CLASSIFICATION
    # --------------------------------------------------------

    st.header(
        "🚦 Risk Classification"
    )

    risk_table = pd.DataFrame({

        "Risk Level": [

            "LOW RISK",

            "MEDIUM RISK",

            "HIGH RISK"
        ],

        "Risk Score": [

            "< 30",

            "30 – 59.99",

            "≥ 60"
        ],

        "Recommendation": [

            "Normal Processing",

            "Review Transaction",

            "Investigate Transaction"
        ]
    })

    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "The risk thresholds are application-level "
        "heuristics for this prototype and are not "
        "bank-approved fraud thresholds."
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.header(
        "📚 Dataset"
    )

    st.write(
        "FraudShield was developed using the "
        "IEEE-CIS Fraud Detection dataset."
    )

    st.write(
        "The dataset contains transaction and identity "
        "signals, including many anonymized features."
    )

    st.warning(
        "⚠️ Several IEEE-CIS fields are anonymized. "
        "FraudShield therefore does not claim that those "
        "fields have the same meaning as real-world bank "
        "fields."
    )

    st.divider()

    # --------------------------------------------------------
    # REAL-WORLD DEPLOYMENT
    # --------------------------------------------------------

    st.header(
        "🔐 Real-World Deployment"
    )

    st.write(
        """
        A production banking fraud detection system would
        require bank-specific historical transaction data,
        secure payment infrastructure, model calibration,
        continuous monitoring, fraud feedback loops,
        drift detection and strict security controls.
        """
    )

    st.info(
        "FraudShield is an ML portfolio and research "
        "prototype, not a production banking fraud system."
    )