import joblib
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from Helpers.IOdata import IO
from Helpers.plotters import PLOT
from Definitions.constants import OUTPUT_DIR_dt
from Helpers.results_logger import ResultsLogger

class DT:
    def __init__(self):
        self.io = IO()
        self.plt = PLOT()
        self.scalar = StandardScaler()

    def decision_tree_model(self):
        Dt = DecisionTreeClassifier(random_state=42)

        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }


        grid = GridSearchCV(estimator=Dt,
                            param_grid=param_grid,
                            cv=5, verbose=True,
                            n_jobs=-1,
                            scoring="accuracy"
                            )

        X_train = self.scalar.fit_transform(self.io.X_train)
        X_test = self.scalar.transform(self.io.X_test)

        grid.fit(X_train, self.io.y_train)

        prediction = grid.predict(X_test)

        # Initialize results logger
        logger = ResultsLogger("DecisionTree")
        
        # Log results to file
        logger.log_results(
            best_params=grid.best_params_,
            best_score=grid.best_score_,
            classification_report_text=classification_report(self.io.y_test, prediction),
            confusion_matrix=confusion_matrix(self.io.y_test, prediction)
        )

        joblib.dump(grid.best_estimator_, "ML_Model/DT/decision_tree.pkl")

        value = confusion_matrix(y_true=self.io.y_test, y_pred=prediction)
        self.plt.plot_confusion_matrix(value=value, x=OUTPUT_DIR_dt)

        return None
