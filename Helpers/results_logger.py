from pathlib import Path
from datetime import datetime
from Definitions.constants import RESULTS_DIR


class ResultsLogger:
    def __init__(self, model_name):
        self.model_name = model_name
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.results_dir / f"{model_name}_results.txt"
        
    def log_results(self, best_params, best_score, classification_report_text, confusion_matrix=None):
        """Log model training results to a text file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"MODEL: {self.model_name.upper()}\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("BEST HYPERPARAMETERS:\n")
            f.write("-" * 40 + "\n")
            if best_params:
                for param, value in best_params.items():
                    f.write(f"{param}: {value}\n")
            else:
                f.write("No hyperparameter tuning performed\n")
            f.write("\n")
            
            f.write("BEST CROSS-VALIDATION SCORE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {best_score:.4f}\n")
            f.write("\n")
            
            f.write("CLASSIFICATION REPORT:\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report_text)
            f.write("\n")
            
            if confusion_matrix is not None:
                f.write("CONFUSION MATRIX:\n")
                f.write("-" * 40 + "\n")
                f.write(str(confusion_matrix))
                f.write("\n\n")
                
                # Calculate and display confusion matrix metrics
                tn, fp, fn, tp = confusion_matrix.ravel()
                f.write("Confusion Matrix Metrics:\n")
                f.write(f"True Negatives: {tn}\n")
                f.write(f"False Positives: {fp}\n")
                f.write(f"False Negatives: {fn}\n")
                f.write(f"True Positives: {tp}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Results saved to: {self.log_file}")
