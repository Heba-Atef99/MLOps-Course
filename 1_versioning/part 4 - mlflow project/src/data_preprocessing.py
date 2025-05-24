import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import logging

RANDOM_SEED = 6

def load_and_preprocess_data(file_path: Path):
    logging.info("Loading dataset...")
    dataset = pd.read_csv(file_path)
    
    # Identify column types
    numerical_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Loan_Status')
    categorical_cols.remove('Loan_ID')

    # Train-test split before any transformation
    X = dataset.drop(columns=['Loan_Status', 'Loan_ID'])
    y = dataset['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    # === Impute categorical features ===
    from sklearn.impute import SimpleImputer

    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    # === Impute numerical features ===
    num_imputer = SimpleImputer(strategy='median')
    X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

    # === Clip outliers (based on train quantiles) ===
    lower = X_train[numerical_cols].quantile(0.05)
    upper = X_train[numerical_cols].quantile(0.95)
    X_train[numerical_cols] = X_train[numerical_cols].clip(lower=lower, upper=upper, axis=1)
    X_test[numerical_cols] = X_test[numerical_cols].clip(lower=lower, upper=upper, axis=1)

    # === Feature engineering: log(LoanAmount), TotalIncome ===
    X_train['LoanAmount'] = np.log1p(X_train['LoanAmount'])
    X_test['LoanAmount'] = np.log1p(X_test['LoanAmount'])

    X_train['TotalIncome'] = X_train['ApplicantIncome'] + X_train['CoapplicantIncome']
    X_test['TotalIncome'] = X_test['ApplicantIncome'] + X_test['CoapplicantIncome']

    X_train['TotalIncome'] = np.log1p(X_train['TotalIncome'])
    X_test['TotalIncome'] = np.log1p(X_test['TotalIncome'])

    # Drop original income columns
    X_train = X_train.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])
    X_test = X_test.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

    # === Label encode categorical variables ===
    cat_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        cat_encoders[col] = le  # Save encoder if needed later

    # === Encode target variable ===
    target_le = LabelEncoder()
    y_train = target_le.fit_transform(y_train)
    y_test = target_le.transform(y_test)
    return X_train, X_test, y_train, y_test