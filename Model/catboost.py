# CatBoost Model Implementation


from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from Helpers.plotters import PLOT
from Helpers.results_logger import ResultsLogger
from Definitions.constants import OUTPUT_DIR_cb


class CatBoost:
    def __init__(self, io=None):
        if io is None:
            from Helpers.IOdata import IO
            self.io = IO()
        else:
            self.io = io
        self.plt = PLOT()
        
    def train(self):
        catboost = CatBoostClassifier()
        param_grid = {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [3, 5, 7],
            'l2_leaf_reg': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bylevel': [0.8, 0.9, 1.0]
        }

        grid = GridSearchCV(catboost, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(self.io.X_train, self.io.y_train)

        print(grid.best_params_)

        prediction = grid.predict(self.io.X_test)

        # Initialize results logger
        logger = ResultsLogger("catboost")
        
        # Log results to file
        logger.log_results(
            best_params=grid.best_params_,
            best_score=grid.best_score_,
            classification_report_text=classification_report(self.io.y_test, prediction),
            confusion_matrix=confusion_matrix(self.io.y_test, prediction)
        )

        print(classification_report(y_true=self.io.y_test, y_pred=prediction))
        joblib.dump(grid.best_estimator_, "ML_Model/CAT/catboost.pkl")

        value = confusion_matrix(y_true=self.io.y_test, y_pred=prediction)
        self.plt.plot_confusion_matrix(value=value, x=OUTPUT_DIR_cb)

        return None