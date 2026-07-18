import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from Definitions.constants import OUTPUT_DIR_lr
from Helpers.IOdata import IO
from Helpers.plotters import PLOT


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

        print(classification_report(y_true=self.io.y_test, y_pred=prediction))
        value = confusion_matrix(y_true=self.io.y_test, y_pred=prediction)

        self.plot.plot_confusion_matrix(value=value, x=OUTPUT_DIR_lr)

        joblib.dump(lr, "ML_Model/LR/lr.pkl")

        return None
