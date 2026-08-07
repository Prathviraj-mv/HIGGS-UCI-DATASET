# Support Vector Machine Model Implementation

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from Helpers.plotters import PLOT
from Definitions.constants import OUTPUT_DIR_svm

class SVM:
    def __init__(self, io=None):
        if io is None:
            from Helpers.IOdata import IO
            self.io = IO()
        else:
            self.io = io
        self.plt = PLOT()
        
    def train(self):
        svm = SVC()
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }

        grid = GridSearchCV(svm,
                            param_grid,
                            cv=5,
                            scoring='accuracy',
                            n_jobs=-1)

        grid.fit(self.io.X_train, self.io.y_train)

        print(grid.best_params_)

        prediction = grid.predict(self.io.X_test)


        print(classification_report(y_true=self.io.y_test, y_pred=prediction))
        joblib.dump(grid.best_estimator_, "ML_Model/SVM/svm.pkl")

        value = confusion_matrix(y_true=self.io.y_test, y_pred=prediction)
        self.plt.plot_confusion_matrix(value=value, x=OUTPUT_DIR_svm)

        return None