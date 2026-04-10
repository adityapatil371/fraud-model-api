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
import mlflow

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
        ('model', XGBClassifier(n_estimators=10, scale_pos_weight=scale_pos_weight))
    ])

    return pipeline

def evaluate(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "auc_roc": roc_auc_score(y_val, y_prob),
        "auc_pr": average_precision_score(y_val, y_prob)
    }
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    return metrics

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
    scale_pos_weight = pos_weight[0] / pos_weight[1]
    
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run():
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        pipeline = build_pipeline(scale_pos_weight)
        best_model = train(X_train, y_train, pipeline)

        mlflow.log_params(best_model.best_params_)

        metrics = evaluate(best_model, X_val, y_val)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(best_model, "model", registered_model_name="fraud-detection-model")

if __name__ == '__main__':
    main()

