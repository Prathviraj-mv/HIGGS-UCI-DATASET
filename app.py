from Helpers.data_extract_and_preprocessing import DATA
from Model.xgb_boost import XGB_MODEL
from Model.Random_forest import RF_MODEL
from Model.logistic_regression import LR_MODEL
from Model.ada_boost import AdaBoost
# from Model.catboost import CatBoost
# from Model.lightgbm import LightGBM

from Model.ensemble import Ensemble
from Model.knn import KNN
from Model.Decision_tree import DT
from EDA.eda_analysis import EDA
from Model.svm import SVM



class APP_:
    def __init__(self):
        print("KNN AWAY")
        # EDA().run()
        DATA().return_data()
        KNN().train()
        print("KNN DONE")

        LR_MODEL().lr_model()
        XGB_MODEL().xgb_model()
        RF_MODEL().rf_model()
        DT().decision_tree_model()
        AdaBoost().train()
        # CatBoost().train()
        Ensemble().train()
        # LightGBM().train()
        SVM().train()


if __name__ == "__main__":
    a = APP_()



