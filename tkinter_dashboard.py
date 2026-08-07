import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import os
import subprocess
import platform
from datetime import datetime

# Import model classes
from Model.xgb_boost import XGB_MODEL
from Model.Random_forest import RF_MODEL
from Model.logistic_regression import LR_MODEL
from Model.ada_boost import AdaBoost
from Model.ensemble import Ensemble
from Model.knn import KNN
from Model.Decision_tree import DT
from Model.svm import SVM
try:
    from Model.lightgbm import LightGBM
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    
try:
    from Model.catboost import CatBoost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    
try:
    from Model.neural_network import NeuralNetwork
    NEURAL_NETWORK_AVAILABLE = True
except ImportError:
    NEURAL_NETWORK_AVAILABLE = False
    
try:
    from Model.naive_bayes import NaiveBayes
    NAIVE_BAYES_AVAILABLE = True
except ImportError:
    NAIVE_BAYES_AVAILABLE = False


class ModelTrainingThread(threading.Thread):
    def __init__(self, models_to_train, progress_queue, result_queue):
        super().__init__()
        self.models_to_train = models_to_train
        self.progress_queue = progress_queue
        self.result_queue = result_queue
        self.daemon = True
        
    def run(self):
        for model_name, model_class, method_name in self.models_to_train:
            try:
                self.progress_queue.put(("status", f"Training {model_name}..."))
                
                # Instantiate and train model
                model_instance = model_class()
                model_method = getattr(model_instance, method_name)
                model_method()
                
                self.result_queue.put((model_name, "Success", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                self.progress_queue.put(("status", f"{model_name} completed successfully!"))
                
            except Exception as e:
                self.result_queue.put((model_name, f"Error: {str(e)}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                self.progress_queue.put(("status", f"{model_name} failed: {str(e)}"))
        
        self.progress_queue.put(("done", len(self.models_to_train)))


class HiggsDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("HIGGS Boson Classification Dashboard")
        self.root.geometry("900x700")
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Create queues for thread communication
        self.progress_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Model configuration
        self.models_config = [
            ("XGBoost", XGB_MODEL, "xgb_model"),
            ("Random Forest", RF_MODEL, "rf_model"),
            ("Logistic Regression", LR_MODEL, "lr_model"),
            ("AdaBoost", AdaBoost, "train"),
            ("KNN", KNN, "train"),
            ("Decision Tree", DT, "decision_tree_model"),
            ("SVM", SVM, "train"),
            ("Ensemble", Ensemble, "train")
        ]
        
        # Add optional models if available
        if LIGHTGBM_AVAILABLE:
            self.models_config.append(("LightGBM", LightGBM, "train"))
        if CATBOOST_AVAILABLE:
            self.models_config.append(("CatBoost", CatBoost, "train"))
        if NEURAL_NETWORK_AVAILABLE:
            self.models_config.append(("Neural Network", NeuralNetwork, "train"))
        if NAIVE_BAYES_AVAILABLE:
            self.models_config.append(("Naive Bayes", NaiveBayes, "train"))
        
        self.model_vars = {}
        self.create_widgets()
        
        # Start checking queues
        self.root.after(100, self.check_queues)
        
    def open_directory(self, path):
        """Open directory in the system's file manager"""
        if not os.path.exists(path):
            messagebox.showerror("Directory Not Found", f"The directory '{path}' does not exist.")
            return
            
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", path])
            else:  # Linux
                subprocess.run(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open directory: {str(e)}")
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="⚛️ HIGGS Boson Classification Dashboard", 
                               font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Model selection
        left_panel = ttk.LabelFrame(main_frame, text="Model Selection", padding="10")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model checkboxes
        for idx, (model_name, _, _) in enumerate(self.models_config):
            var = tk.BooleanVar()
            self.model_vars[model_name] = var
            checkbox = ttk.Checkbutton(left_panel, text=model_name, variable=var)
            checkbox.grid(row=idx, column=0, sticky=tk.W, pady=2)
        
        # Select/Deselect buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=len(self.models_config), column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(button_frame, text="Select All", command=self.select_all).grid(row=0, column=0, padx=2)
        ttk.Button(button_frame, text="Deselect All", command=self.deselect_all).grid(row=0, column=1, padx=2)
        
        # Training button
        self.train_button = ttk.Button(left_panel, text="🚀 Train Selected Models", 
                                      command=self.start_training, style='Accent.TButton')
        self.train_button.grid(row=len(self.models_config) + 1, column=0, columnspan=2, pady=(20, 0), sticky=tk.EW)
        
        # Progress bar
        self.progress = ttk.Progressbar(left_panel, mode='indeterminate')
        self.progress.grid(row=len(self.models_config) + 2, column=0, columnspan=2, pady=(10, 0), sticky=tk.EW)
        
        # Status label
        self.status_label = ttk.Label(left_panel, text="Ready", wraplength=200)
        self.status_label.grid(row=len(self.models_config) + 3, column=0, columnspan=2, pady=(5, 0))
        
        # Right panel - Results
        right_panel = ttk.LabelFrame(main_frame, text="Training Results", padding="10")
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(right_panel, width=50, height=20, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Clear results button
        ttk.Button(right_panel, text="🗑️ Clear Results", command=self.clear_results).grid(row=1, column=0, pady=(10, 0))
        
        # Bottom panel - Additional features
        bottom_panel = ttk.LabelFrame(main_frame, text="Additional Features", padding="10")
        bottom_panel.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(bottom_panel, text="📁 Open Models Folder", command=self.view_saved_models).grid(row=0, column=0, padx=5)
        ttk.Button(bottom_panel, text="📈 Open Results Folder", command=self.view_results_files).grid(row=0, column=1, padx=5)
        ttk.Button(bottom_panel, text="🖼️ Open Plots Folder", command=self.view_plots).grid(row=0, column=2, padx=5)
        
    def select_all(self):
        for var in self.model_vars.values():
            var.set(True)
            
    def deselect_all(self):
        for var in self.model_vars.values():
            var.set(False)
            
    def start_training(self):
        # Get selected models
        selected_models = []
        for model_name, model_class, method_name in self.models_config:
            if self.model_vars[model_name].get():
                selected_models.append((model_name, model_class, method_name))
        
        if not selected_models:
            messagebox.showwarning("No Models Selected", "Please select at least one model to train.")
            return
        
        # Disable button and start progress
        self.train_button.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Training in progress...")
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Starting training for {len(selected_models)} model(s)...\n\n")
        
        # Start training thread
        self.training_thread = ModelTrainingThread(selected_models, self.progress_queue, self.result_queue)
        self.training_thread.start()
        
    def check_queues(self):
        # Check progress queue
        try:
            while True:
                msg_type, message = self.progress_queue.get_nowait()
                if msg_type == "status":
                    self.status_label.config(text=message)
                    self.results_text.insert(tk.END, message + "\n")
                    self.results_text.see(tk.END)
                elif msg_type == "done":
                    self.progress.stop()
                    self.train_button.config(state=tk.NORMAL)
                    self.status_label.config(text=f"Training completed! Processed {message} model(s).")
                    self.results_text.insert(tk.END, f"\n✅ Training completed! Processed {message} model(s).\n")
                    self.results_text.see(tk.END)
        except queue.Empty:
            pass
        
        # Check result queue
        try:
            while True:
                model_name, status, timestamp = self.result_queue.get_nowait()
                if "Error" in status:
                    self.results_text.insert(tk.END, f"❌ {model_name}: {status} ({timestamp})\n")
                else:
                    self.results_text.insert(tk.END, f"✅ {model_name}: {status} ({timestamp})\n")
                self.results_text.see(tk.END)
        except queue.Empty:
            pass
        
        # Continue checking
        self.root.after(100, self.check_queues)
        
    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.status_label.config(text="Ready")
        
    def view_saved_models(self):
        model_dirs = ["ML_Model/XGBoost", "ML_Model/RF", "ML_Model/LR", "ML_Model/KNN", 
                     "ML_Model/DT", "ML_Model/SVM", "ML_Model/LGB", "ML_Model/CB",
                     "ML_Model/NN", "ML_Model/ADA", "ML_Model/NB", "ML_Model/ENS"]
        
        model_files = []
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                model_files.extend([f"{model_dir}/{f}" for f in files if f.endswith('.pkl')])
        
        if model_files:
            # Open the main ML_Model directory
            self.open_directory("ML_Model")
        else:
            messagebox.showwarning("No Models", "No saved models found. Train models first.")
            
    def view_results_files(self):
        results_dir = "Results"
        if os.path.exists(results_dir):
            files = os.listdir(results_dir)
            if files:
                # Open the Results directory
                self.open_directory(results_dir)
            else:
                messagebox.showwarning("No Results", "No result files found.")
        else:
            messagebox.showwarning("No Directory", "Results directory does not exist.")
            
    def view_plots(self):
        plots_dir = "Plots"
        if os.path.exists(plots_dir):
            subdirs = [d for d in os.listdir(plots_dir) if os.path.isdir(os.path.join(plots_dir, d))]
            if subdirs:
                # Open the Plots directory
                self.open_directory(plots_dir)
            else:
                messagebox.showwarning("No Plots", "No plot directories found.")
        else:
            messagebox.showwarning("No Directory", "Plots directory does not exist.")


def main():
    root = tk.Tk()
    app = HiggsDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()