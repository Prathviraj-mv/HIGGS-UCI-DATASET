from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
from Helpers.IOdata import IO
from Helpers.plotters import PLOT


class RF_MODEL:
    def __init__(self):
        pass

    def rf_model(self):
        rf = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }

        grid = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring="r2",
            n_jobs=-1,
            verbose=2
        )
        io = IO()

        grid.fit(io.X_train, io.y_train)

        prediction = grid.best_estimator_.predict(io.X_test)

        print(classification_report(y_true=io.y_test,y_pred=prediction))
        value =confusion_matrix(y_true=io.y_test,y_pred=prediction)
        plot =PLOT()
        plot.plot_confusion_matrix(value=value)

        best_model = grid.best_estimator_

        print()
        joblib.dump(best_model,"ML_Model/RF/rfr.pkl")

        return None
