import os
import sys
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path: str, obj) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_models(X_train, y_train, X_test, y_test, models: dict) -> dict:
    """
    Returns dict: {model_name: test_r2_score}
    """
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys) from e
