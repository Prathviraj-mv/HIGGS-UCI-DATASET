import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from Definitions.constants import OUTPUT_DIR_lr
from Helpers.IOdata import IO
from Helpers.plotters import PLOT
from Helpers.results_logger import ResultsLogger


# logistic regression


class LR_MODEL:
    def __init__(self):
        self.io = IO()
        self.plot = PLOT()

    def lr_model(self):
        scaler = StandardScaler()

        X_train = scaler.fit_transform(self.io.X_train)
        X_test = scaler.transform(self.io.X_test)

        param_grid = {
            "C": [0.01, 0.1, 1, 5, 10],
            "solver": ["lbfgs", "saga"],
            "max_iter": [1000, 2000]
        }

        lr = GridSearchCV(
            LogisticRegression(random_state=42,
                               max_iter=2000,
                               ),
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        lr.fit(X_train, self.io.y_train)

        prediction = lr.predict(X_test)

        # Initialize results logger
        logger = ResultsLogger("LogisticRegression")
        
        # Log results to file
        logger.log_results(
            best_params=lr.best_params_,
            best_score=lr.best_score_,
            classification_report_text=classification_report(self.io.y_test, prediction),
            confusion_matrix=confusion_matrix(self.io.y_test, prediction)
        )

        value = confusion_matrix(y_true=self.io.y_test, y_pred=prediction)

        self.plot.plot_confusion_matrix(value=value, x=OUTPUT_DIR_lr)

        joblib.dump(lr, "ML_Model/LR/lr.pkl")

        return None
