import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from Definitions.constants import OUTPUT_DIR_xg
from Helpers.IOdata import IO
from Helpers.plotters import PLOT


class XGB_MODEL:
    def __init__(self):
        self.io = IO()
        self.plot = PLOT()


    def xgb_model(self):
        xgb = XGBClassifier(
            tree_method="hist",
            device="cuda",  # Use NVIDIA GPU
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )

        param_grid = {
            "n_estimators": [200, 300],
            "max_depth": [6, 8],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "gamma": [0, 0.1],
            "min_child_weight": [1, 3]
        }
        grid = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2
        )


        grid.fit(self.io.X_train, self.io.y_train)

        prediction = grid.best_estimator_.predict(self.io.X_test)

        print("\nBest Parameters:")
        print(grid.best_params_)

        print("\nBest Cross Validation Accuracy:")
        print(grid.best_score_)

        print("\nClassification Report")
        print(classification_report(self.io.y_test, prediction))

        value = confusion_matrix(self.io.y_test, prediction)


        self.plot.plot_confusion_matrix(value=value, x=OUTPUT_DIR_xg)  # directory for the plot to be saved

        best_model = grid.best_estimator_

        joblib.dump(best_model, "ML_Model/XGBoost/xgb_model.pkl")

        return None
