import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import shap
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import RandomizedSearchCV

def load_data(path):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pd.read_csv(os.path.join(BASE_DIR, 'data', 'creditcard.csv'))

def split_data(df):
    train, temp = train_test_split(
    df, test_size = 0.3, stratify= df['Class'], random_state = 42
    )

    val, test = train_test_split(
        temp, test_size = 0.5, stratify= temp['Class'], random_state = 42
    )
    return train.drop(columns=['Class']), val.drop(columns=['Class']), test.drop(columns=['Class']), train['Class'], val['Class'], test['Class']

def build_pipeline(scale_pos_weight):
    scale_features = ['Amount', 'Time']
    passthrough_features = [f'V{i}' for i in range(1, 29)]

    preprocessor = ColumnTransformer(transformers=[
        ('scale', StandardScaler(), scale_features),
        ('pass', 'passthrough', passthrough_features)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(n_estimators=10, scale_pos_weight=scale_pos_weight, random_state=42))
    ])

    return pipeline

def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print(f"Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred):.4f}")
    print(f"F1:        {f1_score(y_val, y_pred):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_val, y_prob):.4f}")
    print(f"AUC-PR:    {average_precision_score(y_val, y_prob):.4f}")

def train(X_train, y_train, pipeline):
    param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.3],
    'model__colsample_bytree': [0.6, 0.8, 1.0]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring='average_precision',
        cv=3,
        random_state=42,
        verbose=1
    )

    search.fit(X_train, y_train)
    return search

def save_model(model, path):
    joblib.dump(model, path)

def main():
    df = load_data('')
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    pos_weight = y_train.value_counts()
    scale_pos_weight = pos_weight[0]/pos_weight[1]
    print(scale_pos_weight)
    pipeline = build_pipeline(scale_pos_weight)
    best_model = train(X_train, y_train, pipeline)
    evaluate(best_model, X_val, y_val)
    save_model(best_model, 'model/best_model.pkl')

if __name__ == '__main__':
    main()

