# AdaBoost Model Implementation


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from Helpers.plotters import PLOT
from Helpers.results_logger import ResultsLogger


class AdaBoost:
    def __init__(self, io):
        self.io = io
        self.plt = PLOT()
        
    def train(self):
        adaboost = AdaBoostClassifier()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        }

        grid = GridSearchCV(adaboost,
                    param_grid, 
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1)
        grid.fit(self.io.X_train, self.io.y_train)

        print(grid.best_params_)

        prediction = grid.predict(self.io.X_test)

        # Initialize results logger
        logger = ResultsLogger("adaboost")
        
        # Log results to file
        logger.log_results(
            best_params=grid.best_params_,
            best_score=grid.best_score_,
            classification_report_text=classification_report(self.io.y_test, prediction),
            confusion_matrix=confusion_matrix(self.io.y_test, prediction)
        )

        print(classification_report(y_true=self.io.y_test, y_pred=prediction))
        joblib.dump(grid.best_estimator_, "ML_Model/ADA/ada.pkl")
    
        value = confusion_matrix(y_true=self.io.y_test, y_pred=prediction)
        self.plt.plot_confusion_matrix(value=value, x=OUTPUT_DIR_adaboost)

        return None