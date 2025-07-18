import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import warnings
import joblib
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

train_df = pd.read_csv("preprocessed_dataset/train_df.csv")
test_df = pd.read_csv("preprocessed_dataset/test_df.csv")

X_train = train_df.drop("Personality", axis=1)
X_test = test_df.drop("Personality", axis=1)
y_train = train_df["Personality"]
y_test = test_df["Personality"] 

with mlflow.start_run(run_name="Random Forest with GridSearchCV"):
    # Define the model
    rf = RandomForestClassifier(random_state=42)

    # Parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    mlflow.log_dict(param_grid, "param_grid.json")

    # Log dataset information
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("n_classes", len(np.unique(y_train)))

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_rf_model = grid_search.best_estimator_

    # Log parameters of the best model
    mlflow.log_params(grid_search.best_params_)

    # Make predictions with the best model
    y_pred = best_rf_model.predict(X_test)
    y_proba = best_rf_model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    mlflow.log_dict(cm.tolist(), "confusion_matrix.json")

    # SOLUSI 1: Simpan model menggunakan joblib dan log sebagai artifact
    model_filename = "random_forest_model.pkl"
    joblib.dump(best_rf_model, model_filename)
    mlflow.log_artifact(model_filename, "model")
    
    # Hapus file temporary
    os.remove(model_filename)

    # Log feature importance
    feature_importance = best_rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
    feature_importance_df.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")
    
    # Hapus file temporary
    os.remove("feature_importance.csv")

    # Log Tags
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("framework", "sklearn")
    mlflow.set_tag("model_format", "joblib")
    
    print(f"Experiment completed successfully!")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")