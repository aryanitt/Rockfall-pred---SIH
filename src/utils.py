import sys
import os
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.exception_config import CustomException


def save_object(file_path, obj):
    """
    Save Python object into a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    """
    Evaluate multiple classification models with hyperparameter tuning.
    Returns a dictionary with model name and accuracy score.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3, scoring="accuracy", n_jobs=-1)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_test_pred = model.predict(x_test)

            # Collect multiple metrics
            accuracy = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, average="weighted")
            precision = precision_score(y_test, y_test_pred, average="weighted")
            recall = recall_score(y_test, y_test_pred, average="weighted")

            report[model_name] = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load Python object from file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
