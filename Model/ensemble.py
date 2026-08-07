# Ensemble Methods (Voting/Stacking) Implementation
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from Helpers.plotters import PLOT
from Helpers.results_logger import ResultsLogger
from Definitions.constants import OUTPUT_DIR_ensemble


class Ensemble:
    def __init__(self, io=None):
        if io is None:
            from Helpers.IOdata import IO
            self.io = IO()
        else:
            self.io = io
        self.plt = PLOT()
        
    def train(self):

        voting = VotingClassifier([
            ('lr', LogisticRegression()),
            ('dt', DecisionTreeClassifier()),
            ('svm', SVC())
        ])

        stacking = StackingClassifier([
            ('lr', LogisticRegression()),
            ('dt', DecisionTreeClassifier()),
            ('svm', SVC())

        ], final_estimator=LogisticRegression())
        param_grid = {
            'voting': ['hard', 'soft'],
            'stacking': [True, False]
        }
        
        grid = GridSearchCV(voting,
         param_grid, 
         cv=5,
          scoring='accuracy', 
          n_jobs=-1)

        grid.fit(self.io.X_train, self.io.y_train)
        print(grid.best_params_)

        prediction = grid.predict(self.io.X_test)

        print(classification_report(y_true=self.io.y_test, y_pred=prediction))

        joblib.dump(grid.best_estimator_, "ML_Model/ENSEMBLE/ensemble.pkl")
        
        # Initialize results logger
        logger = ResultsLogger("ensemble")
        
        # Log results to file
        logger.log_results(
            best_params=grid.best_params_,
            best_score=grid.best_score_,
            classification_report_text=classification_report(self.io.y_test, prediction),
            confusion_matrix=confusion_matrix(self.io.y_test, prediction)
        )


        value = confusion_matrix(y_true=self.io.y_test, y_pred=prediction)

        self.plt.plot_confusion_matrix(value=value, x=OUTPUT_DIR_ensemble)

        return None