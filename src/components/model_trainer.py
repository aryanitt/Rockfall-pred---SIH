import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, f1_score

from src.exception_config import CustomException
from src.logger_config import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting independent and dependent features")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1], test_array[:, -1]
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=500),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "XGB Classifier": XGBClassifier(eval_metric="logloss"),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
                "CatBoost Classifier": CatBoostClassifier(verbose=False, random_state=42)
            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1.0, 10],
                    "solver": ["liblinear", "lbfgs"]
                },
                "K-Neighbors Classifier": {
                    "n_neighbors": [3, 5, 10],
                    "weights": ["uniform", "distance"]
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5]
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None]
                },
                "XGB Classifier": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1]
                },
                "AdaBoost Classifier": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1]
                },
                "CatBoost Classifier": {
                    "iterations": [200, 500],
                    "learning_rate": [0.01, 0.05]
                }
            }

            # Evaluate models
            model_report: dict = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Print for debugging (optional)
            logging.info(f"Model report: {model_report}")

            # Robust best model selection
            def get_score(value):
                if isinstance(value, dict):
                    # Try common keys
                    for key in ['score', 'accuracy', 'f1_score']:
                        if key in value:
                            return value[key]
                    # If no key found, take the first numeric value
                    for v in value.values():
                        if isinstance(v, (int, float)):
                            return v
                    raise ValueError(f"No numeric score found in {value}")
                elif isinstance(value, (int, float)):
                    return value
                else:
                    raise ValueError(f"Unexpected value type: {type(value)}")

            best_model_name = max(model_report, key=lambda k: get_score(model_report[k]))
            best_model_score = get_score(model_report[best_model_name])
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with score {best_model_score}")

            # Train the best model
            best_model.fit(x_train, y_train)

            # Save trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Evaluate on test set
            predicted = best_model.predict(x_test)
            acc = accuracy_score(y_test, predicted)
            f1 = f1_score(y_test, predicted, average="weighted")  # handles multiclass

            return {
                "best_model": best_model_name,
                "accuracy": acc,
                "f1_score": f1
            }

        except Exception as e:
            raise CustomException(e, sys)
