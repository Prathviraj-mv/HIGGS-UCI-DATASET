# Neural Network (MLP) Model Implementation

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from Helpers.plotters import PLOT
from Helpers.results_logger import ResultsLogger
from Definitions.constants import OUTPUT_DIR_nn
from Helpers.IOdata import IO

class NeuralNetwork:
    def __init__(self, io=None):
        if io is None:
            self.io = IO()
        else:
            self.io = io
        self.plt = PLOT()
        
    def train(self):
        mlp = MLPClassifier(random_state=42, max_iter=1000)
        
        param_grid = {
            'hidden_layer_sizes': [(100,), (100, 50), (50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive']
        }
        
        grid = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid.fit(self.io.X_train, self.io.y_train)
        
        print(grid.best_params_)
        
        prediction = grid.predict(self.io.X_test)
        
        # Initialize results logger
        logger = ResultsLogger("NeuralNetwork")
        
        # Log results to file
        logger.log_results(
            best_params=grid.best_params_,
            best_score=grid.best_score_,
            classification_report_text=classification_report(self.io.y_test, prediction),
            confusion_matrix=confusion_matrix(self.io.y_test, prediction)
        )
        
        print(classification_report(y_true=self.io.y_test, y_pred=prediction))
        joblib.dump(grid.best_estimator_, "ML_Model/NN/neural_network.pkl")
        
        value = confusion_matrix(y_true=self.io.y_test, y_pred=prediction)
        self.plt.plot_confusion_matrix(value=value, x=OUTPUT_DIR_nn)
        
        return None
