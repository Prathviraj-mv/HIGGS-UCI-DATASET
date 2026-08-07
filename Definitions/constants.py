from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / "Dataset" / "HIGGS_500k.csv"

RESULTS_DIR = ROOT_DIR / "Results"

MODELS_DIR = ROOT_DIR / "ML_Model"

OUTPUT_DIR_p = ROOT_DIR / "EDA Plots"
OUTPUT_DIR_rf = ROOT_DIR / "Plots" / "RF"
OUTPUT_DIR_lr = ROOT_DIR / "Plots" / "LR"
OUTPUT_DIR_xg = ROOT_DIR / "Plots" / "XG"
OUTPUT_DIR_kn = ROOT_DIR / "Plots" / "KN"
OUTPUT_DIR_dt = ROOT_DIR / "Plots" / "DT"
OUTPUT_DIR_lgb = ROOT_DIR / "Plots" / "LGB"
OUTPUT_DIR_cb = ROOT_DIR / "Plots" / "CB"
OUTPUT_DIR_nn = ROOT_DIR / "Plots" / "NN"
OUTPUT_DIR_svm = ROOT_DIR / "Plots" / "SVM"
OUTPUT_DIR_ada = ROOT_DIR / "Plots" / "ADA"
OUTPUT_DIR_adaboost = ROOT_DIR / "Plots" / "ADA"
OUTPUT_DIR_nb = ROOT_DIR / "Plots" / "NB"
OUTPUT_DIR_ens = ROOT_DIR / "Plots" / "ENS"
OUTPUT_DIR_ensemble = ROOT_DIR / "Plots" / "ENS"



