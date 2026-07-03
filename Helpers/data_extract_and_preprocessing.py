import pandas as pd
from Definitions.constants import DATASET_PATH
path_ = DATASET_PATH

class DATA:
    def __init__(self):
        self.path = path_
        self.data = pd.read_csv(self.path)

    def head(self):
        return print(self.data.head(5))

    def print_data_col(self):
        return print(self.data.columns)

    def null_data(self):
        return print(self.data.isnull().sum())

    def calc_corr(self):
        pass

    def print_corr(self):
        pass


    # Final Data Return

    def return_data(self):
        return self.data





if __name__ == "__main__":
    data =DATA()
    data.return_data()
    data.print_data_col()
    data.head()
    data.print_corr()

