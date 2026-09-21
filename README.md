# Credit Card Fraud Detection API

An end-to-end machine learning pipeline that detects fraudulent credit card transactions — from data ingestion through SQL Server, to exploratory analysis, model training, and a production-ready FastAPI service for real-time predictions.

## Business Problem

Financial fraud costs the global economy billions every year. Rule-based detection systems are rigid and struggle to catch evolving fraud patterns. This project builds a machine learning model that flags fraudulent transactions automatically, aiming to **minimize missed fraud (false negatives)** while keeping false alarms (false positives) at a reasonable level.

## Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Records | 284,807 transactions |
| Fraud cases | 492 (only **0.17%** of all transactions) |
| Time period | 2 days, September 2013, European cardholders |
| Features | 30 input features + 1 target (`Class`) |

| Column | Description |
|---|---|
| `Time` | Seconds since the first transaction (converted to hours in this project) |
| `Amount` | Transaction value |
| `V1`–`V28` | PCA-transformed features (anonymized) |
| `Class` | Target — `0` = legitimate, `1` = fraud |

## Pipeline

### 1. Data ingestion via SQL Server
The raw CSV is loaded, pushed into a local SQL Server database with SQLAlchemy, then read back with a SQL query — reflecting a more realistic data-engineering workflow than reading straight from a flat file.

```python
en = create_engine(
    f'mssql+pyodbc://{server_name}/{database_name}'
    f'?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'
)
data.to_sql(name=database_name, chunksize=100, index=False, if_exists='replace', con=en)
data = pd.read_sql("SELECT * FROM [Credit-Card-Fraud-Detection-ML-Pipeline]", con=en)
```

### 2. Exploratory Data Analysis
- No missing values across all 284,807 records.
- Severe class imbalance — fraud is only 0.17% of transactions.
- Fraud is concentrated in specific hours once `Time` is converted from seconds to a 24-hour clock.
- A share of fraud transactions have an `Amount` of exactly 0.
- Fraud transactions above the average amount were analyzed separately for timing patterns.

### 3. Feature engineering
```python
data['Time'] = round(((data['Time'] / 3600) % 24).astype(float), 4)
```
Converting `Time` to hours makes the temporal fraud pattern visible, which is lost in the raw seconds format.

### 4. Missing value handling
```python
Imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = Imputer.fit_transform(X)
```

### 5. Train/test split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.27, random_state=33, shuffle=True
)
```
73% training / 27% test, split **before** scaling to avoid data leakage.

### 6. Feature scaling
```python
standModel = StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = standModel.fit_transform(X_train)   # fit on train only
X_test  = standModel.transform(X_test)        # apply same transform to test
```
The scaler is fit **only** on the training set and then applied to the test set — no leakage of test-set statistics into training.

### 7. Handling class imbalance
```python
over_sample = RandomOverSampler(random_state=33)
X_train_re, y_train_re = over_sample.fit_resample(X_train, y_train)
```
Oversampling is applied to the training set only, after the split, so the test set stays untouched and representative of real-world class distribution.

### 8. Model training
Two classifiers were trained and compared:
- `LogisticRegression(solver='sag')`
- `RandomForestClassifier(n_estimators=100)`

### 9. Evaluation
Each model was evaluated with metrics suited to imbalanced classification: recall, zero-one loss, and the confusion matrix.

## Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Recall (fraud caught) | **90.7%** | 70.9% |
| Zero-one loss | higher | **~0.05%** |
| False positives | 1,971 | **5** |
| False negatives | **13** | 41 |

**Chosen model: Logistic Regression.**

Random Forest is more "quiet" (very few false alarms), but it misses 41 real fraud cases. Logistic Regression catches 128 of 141 fraud cases in the test set — a much better fit for a domain where a missed fraud is far more costly than a false alarm.

### 10. Saving the model
```python
joblib.dump(logistic_model, 'The_Best_Model_prediction_fraud_model.pkl')
joblib.dump(standModel, 'Scaler_Transform.pkl')
```

## API

Served with **FastAPI**, so the trained model can return predictions over HTTP instead of only running inside a script.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Accepts transaction features, returns a fraud prediction |

The API loads `The_Best_Model_prediction_fraud_model.pkl` and `Scaler_Transform.pkl` at startup, using a path resolved relative to `main.py` itself — so it works regardless of which directory the server is launched from. If either file fails to load, the app raises an error immediately at startup rather than failing later on the first request.

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "Time": 0, "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37,
        "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36,
        "V10": 0.09, "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31,
        "V15": 1.46, "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40,
        "V20": 0.25, "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06,
        "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02, "Amount": 149.62
      }'
```

### Example response

```json
{
  "predict": 0,
  "value": "normal_case"
}
```

## Project Structure

```
.
├── main.py                                     # FastAPI app serving the trained model
├── Credit_Card_Fraud_Detection_ML_Pipeline.py  # Full pipeline: SQL ingestion, EDA, training, evaluation
├── The_Best_Model_prediction_fraud_model.pkl   # Saved Logistic Regression model
├── Scaler_Transform.pkl                        # Saved StandardScaler
├── requirements.txt
└── README.md
```

## Getting Started

**Requirements:** Python 3.12.5

1. Clone the repository and move into the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `creditcard.csv` in the project folder if you want to re-run the training pipeline.
4. Update `server_name` and `database_name` in `Credit_Card_Fraud_Detection_ML_Pipeline.py` if you want to reproduce the SQL Server ingestion step (a local SQL Server instance with the ODBC Driver 17 is required).
5. Run the training pipeline (optional, only if retraining):
   ```bash
   python Credit_Card_Fraud_Detection_ML_Pipeline.py
   ```
6. Make sure `The_Best_Model_prediction_fraud_model.pkl` and `Scaler_Transform.pkl` are in the same folder as `main.py`.
7. Run the API:
   ```bash
   uvicorn main:app --reload
   ```
8. Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

## Requirements

```
fastapi==0.141.1
uvicorn==0.52.4
pydantic==2.13.5
joblib==1.4.2
numpy==2.0.2
pandas==2.2.3
scikit-learn==1.5.2
imbalanced-learn==0.14.1
matplotlib==3.10.0
SQLAlchemy==2.0.49
pyodbc
```

## Limitations & Future Improvements

- Reproducing the SQL Server ingestion step requires a local SQL Server instance and the correct ODBC driver, which limits portability; a CSV-only fallback path would make the project easier for others to run.
- The `V1`–`V28` features are anonymized via PCA, limiting interpretability of individual predictions.
- The false-positive rate of the chosen model could be reduced with threshold tuning or cost-sensitive learning.
- No automated tests or CI pipeline yet.
- The training script is currently a single procedural file; splitting it into functions/modules (ingestion, EDA, training, evaluation) would improve maintainability.

## Tech Stack

Python 3.12.5, pandas, scikit-learn, imbalanced-learn, SQLAlchemy, FastAPI, uvicorn, joblib, matplotlib
