# LightGBM Model Implementation

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from Helpers.plotters import PLOT
from lightgbm import LGBMClassifier
from Definitions.constants import OUTPUT_DIR_lgb

class LightGBM:
    def __init__(self, io):
        self.io = io
        self.plt = PLOT()
        
    def train(self):
        lgbm = LGBMClassifier()
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'num_leaves': [31, 50, 100],
            'min_child_samples': [20, 50, 100],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        grid = GridSearchCV(lgbm, param_grid, 
                                cv=5, 
                                scoring='accuracy', 
                                n_jobs=-1)
        grid.fit(self.io.X_train, self.io.y_train)

        print(grid.best_params_)

        prediction = grid.predict(self.io.X_test)


        print(classification_report(y_true=self.io.y_test, y_pred=prediction))
        joblib.dump(grid.best_estimator_, "ML_Model/LGB/lgbm.pkl")

        value = confusion_matrix(y_true=self.io.y_test, y_pred=prediction)
        self.plt.plot_confusion_matrix(value=value, x=OUTPUT_DIR_lgb)

        return None
