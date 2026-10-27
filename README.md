# 🛡️ FraudShield — AI-Powered Fraud Detection System

FraudShield is an end-to-end machine learning project for detecting potentially fraudulent financial transactions using **XGBoost**.

The project combines **machine learning, risk scoring, model explainability, and an interactive Streamlit application** to demonstrate how an AI-based fraud detection system can analyze transactions and highlight suspicious activity.

> ⚠️ **Disclaimer:** FraudShield is an educational and portfolio project. It does not connect to real banking or payment networks and should not be used for real financial decisions.

---

## 🚀 Live Demo

🌐 **Streamlit App:**  
https://fraudshield-4dhscbi9u2uanmbe2cgkvy.streamlit.app/

---

## 🎯 Project Objective

The goal of FraudShield is to build a machine learning system that can:

- Detect potentially fraudulent transactions
- Estimate fraud probability
- Assign a transaction risk level
- Identify important factors influencing predictions
- Provide an interactive interface for analyzing transactions
- Demonstrate model explainability using SHAP

---

## 🤖 Machine Learning Model

FraudShield uses **XGBoost** as the primary fraud detection model.

### Model Performance

| Metric | Score |
|---|---:|
| ROC-AUC | **90.3%** |
| PR-AUC | **58.1%** |

The model is designed to distinguish between legitimate and potentially fraudulent transactions based on transaction-related features.

---

## 🔄 Machine Learning Pipeline

The overall workflow is:

```text
Raw Transaction Data
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
XGBoost Classifier
        ↓
Fraud Probability
        ↓
Risk Analysis
        ↓
SHAP Explainability
        ↓
Streamlit Dashboard
```

---

## 🧠 How FraudShield Works

### 1. Transaction Input

The system receives transaction-related information such as financial and behavioral features.

### 2. Data Processing

The input data is prepared using the same preprocessing approach used during model training.

### 3. XGBoost Prediction

The trained XGBoost model generates a probability indicating how likely the transaction is to be fraudulent.

### 4. Risk Assessment

The predicted probability is converted into an easy-to-understand risk score and risk level.

### 5. Explainability

SHAP is used to identify which features contributed toward increasing or decreasing the model's fraud prediction.

---

# 🚀 Streamlit Application

FraudShield includes an interactive Streamlit application with multiple sections for analyzing and understanding transactions.

## 1. ML Fraud Analysis Dashboard

The dashboard uses the **trained XGBoost model** to estimate fraud probability and provide model-driven risk analysis.

![FraudShield Dashboard](images/dashboard.png)

The dashboard displays:

- Fraud probability
- Risk score
- Risk level
- Prediction result
- Model explanation
- Increased fraud-risk signals
- Reduced fraud-risk signals

---

## 2. Transaction Risk Simulator

The simulator allows users to experiment with different transaction scenarios and observe how various risk signals affect the overall risk score.

### Low-Risk Transaction

![Low-Risk Transaction](images/simulator-low-risk.png)

Example result:

**12.00 / 100 — LOW RISK**

The transaction is classified as suitable for normal processing.

### High-Risk Transaction

![High-Risk Transaction](images/simulator-high-risk.png)

Example result:

**99.00 / 100 — HIGH RISK**

The transaction is flagged for investigation because multiple risk signals are present.

> **Important:** The Transaction Risk Simulator uses a separate transparent **rule-based risk scoring system** for demonstration purposes. It is not the same system as the XGBoost ML prediction shown in the Dashboard.

---

## 3. About FraudShield

The application includes an About section explaining the project objective, machine learning model, performance metrics, and overall pipeline.

![About FraudShield](images/about-model.png)

---

# 🔍 Model Explainability

Fraud detection models can be difficult to interpret. FraudShield therefore uses **SHAP (SHapley Additive exPlanations)** to understand model predictions.

SHAP helps answer:

> **Why did the model make this prediction?**

It identifies which features contributed toward increasing or decreasing the predicted fraud probability.

## SHAP Summary

![SHAP Summary](images/shap_summary.png)

The SHAP summary provides a global view of feature importance across the dataset.

## Individual Fraud Transaction Explanation

![SHAP Fraud Transaction](images/shap_fraud_transaction.png)

This visualization provides an individual explanation of a fraud-related prediction and shows the features contributing to that specific prediction.

---

# 📊 Key Features

### 🤖 Machine Learning

- XGBoost classification model
- Fraud probability prediction
- ROC-AUC and PR-AUC evaluation
- Transaction risk classification

### 🔍 Explainable AI

- SHAP-based model interpretation
- Global feature importance
- Individual prediction explanations
- Increased and reduced risk signals

### 🎨 Interactive Application

- Streamlit web application
- ML prediction dashboard
- Transaction risk simulator
- Risk score visualization
- Model information section

### 🛡️ Risk Analysis

- Low-risk transaction detection
- High-risk transaction detection
- Risk score interpretation
- Transaction investigation recommendations

---

# 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | Machine learning utilities |
| XGBoost | Fraud detection model |
| SHAP | Model explainability |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web application |
| Joblib | Model serialization |
| Jupyter Notebook | Model development |

---

# 📁 Project Structure

```text
FraudShield/
│
├── data/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── models/
│   └── ...
│
├── images/
│   ├── dashboard.png
│   ├── simulator-low-risk.png
│   ├── simulator-high-risk.png
│   ├── about-model.png
│   ├── shap_summary.png
│   └── shap_fraud_transaction.png
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/anantrai3127/FraudShield.git
```

Move into the project directory:

```bash
cd FraudShield
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

# 📈 Model Evaluation

Fraud detection problems are often highly imbalanced, so evaluating the model using only accuracy can be misleading.

FraudShield therefore focuses on metrics such as:

### ROC-AUC

Measures how effectively the model distinguishes fraudulent transactions from legitimate transactions across classification thresholds.

**ROC-AUC: 90.3%**

### PR-AUC

Provides a more informative view of performance when the positive fraud class is relatively rare.

**PR-AUC: 58.1%**

---

# 🔐 Risk Interpretation

The application presents predictions in a simple risk-oriented format.

| Risk Level | Interpretation |
|---|---|
| 🟢 LOW RISK | Transaction appears relatively normal |
| 🟡 MEDIUM RISK | Transaction requires additional attention |
| 🔴 HIGH RISK | Transaction should be investigated |

These categories are intended for demonstration and portfolio purposes.

---

# 💡 What I Learned

Through this project, I practiced:

- Building an end-to-end machine learning project
- Data preprocessing
- Feature engineering
- Training classification models
- Using XGBoost
- Evaluating classification performance
- Handling fraud detection metrics
- Model explainability with SHAP
- Saving and loading trained models
- Building a Streamlit application
- Creating a professional ML project interface
- Deploying a machine learning application
- Structuring a GitHub portfolio project

---

# 🔮 Future Improvements

Possible future improvements include:

- Hyperparameter optimization
- Threshold optimization for fraud detection
- Real-time transaction monitoring
- Improved handling of class imbalance
- More advanced feature engineering
- Model comparison with LightGBM and CatBoost
- Real-time SHAP explanations
- User authentication
- Transaction history and monitoring dashboard
- Model monitoring and drift detection

---

# 👨‍💻 Author

**Anant Rai**

B.Tech — Computer Science & Engineering (AI/ML)

Guru Gobind Singh Indraprastha University (GGSIPU)

### Connect With Me

- GitHub: https://github.com/anantrai3127
- LinkedIn: https://www.linkedin.com/in/anant-rai-205362447/

---

# ⭐ Project

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.