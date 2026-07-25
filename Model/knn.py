# KNN Model Implementation

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from Helpers.plotters import PLOT
from Helpers.results_logger import ResultsLogger
from Definitions.constants import OUTPUT_DIR_kn
from Helpers.IOdata import IO


class KNN:
    def __init__(self):
        self.io = IO()
        self.plt = PLOT()
        
    def train(self):

        knn = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        grid = GridSearchCV(knn, 
                param_grid, 
                cv=5,
                scoring='accuracy', 
                n_jobs=-1)
        grid.fit(self.io.X_train, self.io.y_train)

        print(grid.best_params_)

        prediction = grid.predict(self.io.X_test)


        print(classification_report(y_true=self.io.y_test, y_pred=prediction))
        joblib.dump(grid.best_estimator_, "ML_Model/KNN/knn.pkl")

        value = confusion_matrix(y_true=self.io.y_test, y_pred=prediction)
        self.plt.plot_confusion_matrix(value=value, x=OUTPUT_DIR_kn)

         # Initialize results logger
        logger = ResultsLogger("knn")
        
        # Log results to file
        logger.log_results(
            best_params=grid.best_params_,
            best_score=grid.best_score_,
            classification_report_text=classification_report(self.io.y_test, prediction),
            confusion_matrix=confusion_matrix(self.io.y_test, prediction)
        )

        return None


