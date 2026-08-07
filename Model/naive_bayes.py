# Naive Bayes Model Implementation

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from Helpers.plotters import PLOT
from Helpers.results_logger import ResultsLogger
from Definitions.constants import OUTPUT_DIR_nb
from Helpers.IOdata import IO

class NaiveBayes:
    def __init__(self, io=None):
        if io is None:
            self.io = IO()
        else:
            self.io = io
        self.plt = PLOT()
        
    def train(self):
        nb = GaussianNB()
        
        # GaussianNB has limited hyperparameters, but we can still use GridSearchCV
        param_grid = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
        
        grid = GridSearchCV(nb, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid.fit(self.io.X_train, self.io.y_train)
        
        print(grid.best_params_)
        
        prediction = grid.predict(self.io.X_test)
        
        # Initialize results logger
        logger = ResultsLogger("NaiveBayes")
        
        # Log results to file
        logger.log_results(
            best_params=grid.best_params_,
            best_score=grid.best_score_,
            classification_report_text=classification_report(self.io.y_test, prediction),
            confusion_matrix=confusion_matrix(self.io.y_test, prediction)
        )
        
        print(classification_report(y_true=self.io.y_test, y_pred=prediction))
        joblib.dump(grid.best_estimator_, "ML_Model/NB/naive_bayes.pkl")
        
        value = confusion_matrix(y_true=self.io.y_test, y_pred=prediction)
        self.plt.plot_confusion_matrix(value=value, x=OUTPUT_DIR_nb)
        
        return None
