

> End-to-end Machine Learning pipeline to detect fraudulent credit card transactions —  
> from raw data ingestion via SQL Server to a saved production-ready model using scikit-learn.

---

##  Business Problem

Financial fraud costs the global economy over **$5 trillion annually**.  
Traditional rule-based systems are slow, rigid, and unable to detect sophisticated fraud patterns.

**Goal:** Build a machine learning model that automatically identifies fraudulent transactions in real time — minimizing missed fraud (False Negatives) while keeping false accusations (False Positives) as low as possible.

---

##  Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Records | 284,807 transactions |
| Fraud cases | 492 (only **0.17%**) |
| Time period | 2 days — September 2013, European cardholders |
| Features | 30 input features + 1 target (Class) |

### Columns Overview

| Column | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction (converted to hours in this project) |
| `Amount` | Transaction value in USD |
| `V1 – V28` | PCA-transformed features (anonymized for privacy) |
| `Class` | Target variable — **0 = Legit**, **1 = Fraud** |

---

##  Full Pipeline — Step by Step

### Step 1 — Data Ingestion from SQL Server

Instead of loading data directly from a CSV file, the raw dataset was first imported into **Microsoft SQL Server (SSMS)** and then fetched using Python via `SQLAlchemy` and `pyodbc`.

```python
en = create_engine(
    f'mssql+pyodbc://{server_name}/{database_name}'
    f'?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
)
data = pd.read_sql("SELECT * FROM [Credit-Card-Fraud-Detection-ML-Pipeline]", con=en)
```

 **Why SQL Server?** Working with a database instead of raw CSV files reflects real-world data engineering practices and makes the pipeline more scalable and professional.

---

### Step 2 — Exploratory Data Analysis (EDA)

Before building any model, the data was thoroughly explored to understand its structure and uncover key patterns.

**Key findings:**

-  No missing values detected across all 284,807 records
-  Severe class imbalance: only **0.17%** of transactions are fraudulent
-  Fraud transactions are concentrated in **late-night hours (12 AM – 6 AM)**
-  Some fraud transactions had an **Amount = $0** (card testing behavior)
-  Fraud amounts above the mean tend to cluster around specific hours

```python
# Fraud percentage
print(((data.loc[data['Class']==1, ['Class']].sum()) / (data['Class'].count())) * 100)

# Statistical summary of fraud amounts
fraud_data['Amount'].aggregate(['mean', 'max', 'median', 'min', 'sum'])

# Percentage of zero-amount fraud transactions
((fraud_data.loc[fraud_data['Amount']==0, 'Amount'].count()) / (fraud_data['Amount'].count())) * 100
```

---

### Step 3 — Feature Engineering

**Converting Time from seconds to hours:**

The `Time` column originally represented elapsed seconds since the first transaction.  
It was converted to a 24-hour clock format to enable meaningful temporal analysis.

```python
data['Time'] = round(((data['Time'] / 3600) % 24).astype(float), 4)
```

This transformation revealed that **most fraud occurs between midnight and 6 AM** — a pattern invisible in the raw seconds format.

---

### Step 4 — Handling Missing Values

A `SimpleImputer` was applied to replace any `NaN` values with the column mean — ensuring the dataset is clean before scaling or training.

```python
Imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = Imputer.fit_transform(X)
```

---

### Step 5 — Train / Test Split

The dataset was split **before** applying any scaling to prevent data leakage.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.27, random_state=33, shuffle=True
)
```

| Set | Size |
|---|---|
| Training set | 73% — 207,869 records |
| Test set | 27% — 76,938 records |

---

### Step 6 — Feature Scaling (StandardScaler)

`StandardScaler` was applied **after** the split — fitted only on `X_train`, then used to transform `X_test`.  
This prevents the model from "seeing" test data statistics during training (Data Leakage prevention).

```python
standModel = StandardScaler()
X_train = standModel.fit_transform(X_train)   # Learn from train only
X_test  = standModel.transform(X_test)         # Apply same rules to test
```

The scaler was also saved for future use:
```python
joblib.dump(standModel, 'Scaler_Transform.pkl')
```

---

### Step 7 — Handling Class Imbalance (RandomOverSampler)

With only 0.17% fraud cases, training a model on raw data would result in a biased model that ignores fraud entirely.  
`RandomOverSampler` was used to balance the training set by duplicating minority class samples.

```python
over_sample = RandomOverSampler(random_state=33)
X_train_re, y_train_re = over_sample.fit_resample(X_train, y_train)
```

---

### Step 8 — Model Training

Two classification models were trained and compared:

| Model | Description |
|---|---|
| `LogisticRegression` | Fast, interpretable baseline model |
| `RandomForestClassifier` | Ensemble of 100 decision trees — more powerful |

```python
logistic_model = LogisticRegression(random_state=33, solver='sag')
forest_model   = RandomForestClassifier(random_state=33, n_estimators=100)
```

---

### Step 9 — Evaluation

Models were evaluated using metrics suited for imbalanced datasets:

- **Recall** — How many actual fraud cases were correctly detected?
- **Zero-One Loss** — Overall error rate
- **Confusion Matrix** — Full breakdown of TP, TN, FP, FN

```python
recall     = recall_score(y_test, y_pred)
zero_one   = zero_one_loss(y_test, y_pred, normalize=True)
confusion  = confusion_matrix(y_test, y_pred)
```

---

##  Results & Model Comparison

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **Recall (Fraud detected)** | **90.7%**  | 70.9% |
| **Zero-One Loss** | 2.6% | **0.05%**  |
| **False Positives** | 1,971  | 5  |
| **False Negatives** | 13  | 41  |

### Analysis

**Logistic Regression:**
- Correctly identified **128 out of 141 fraud cases** — excellent recall
- High False Positives (1,971 legitimate transactions flagged as fraud)
- Best choice when **catching all fraud is the priority** — even at the cost of false alarms

**Random Forest:**
- Near-perfect overall accuracy (Zero-One Loss = 0.05%)
- Only **5 false positives** — almost no false alarms
- Missed 41 fraud cases — higher False Negatives
- Best choice when **minimizing false alarms is the priority**

###  Best Model Selected: Logistic Regression
In fraud detection, **missing a real fraud case is far more costly** than a false alarm.  
A Recall of **90.7%** makes Logistic Regression the safer business choice.

---

### Step 10 — Saving the Model

The best-performing model and the scaler were saved as `.pkl` files for future use without retraining.

```python
import joblib
joblib.dump(logistic_model, 'The_Best_Model_prediction_fraud_model.pkl')
joblib.dump(standModel,     'Scaler_Transform.pkl')
```

To load and use the model later:
```python
model  = joblib.load('The_Best_Model_prediction_fraud_model.pkl')
scaler = joblib.load('Scaler_Transform.pkl')

new_data_scaled = scaler.transform(new_data)
prediction      = model.predict(new_data_scaled)
```

---

##  Project Structure

```
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection_ML_Pipeline.py   # Main pipeline script
├── The_Best_Model_prediction_fraud_model.pkl    # Saved Logistic Regression model
├── Scaler_Transform.pkl                         # Saved StandardScaler
└── README.md
```

---

##  How to Run

```bash
# 1. Clone the repository
git clone https://github.com/MohammedAbuSelmia/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your SQL Server connection in the script
#    Update: server_name and database_name variables

# 4. Run the full pipeline
python Credit_Card_Fraud_Detection_ML_Pipeline.py
```

---

##  Requirements

```
pandas
numpy
sqlalchemy
pyodbc
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
```

---

##  Key Insights

| # | Insight |
|---|---|
| 1 |  **Late-night hours (12 AM – 6 AM)** have the highest fraud concentration |
| 2 |  **Zero-amount transactions** exist in fraud data — likely card-testing behavior |
| 3 |  Only **0.17%** of transactions are fraudulent — severe imbalance requires oversampling |
| 4 |  **Recall is the right metric** here — missing fraud is worse than a false alarm |

---
