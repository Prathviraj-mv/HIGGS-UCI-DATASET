import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DATA:
    def __init__(self):

        from Definitions.constants import e_path
        path_ = e_path

        self.path = path_
        self.data = pd.read_csv(self.path)

    def head(self):
        return print(self.data.head(5))

    def print_data_col(self):
        return print(self.data.columns)

    def null_data(self):
        return print(self.data.isnull().sum())

    def fill_null_data(self):
        self.data["Savanna fires"] = self.data["Savanna fires"].fillna(self.data["Savanna fires"].mean())
        self.data["Forest fires"] = self.data["Forest fires"].fillna(self.data["Forest fires"].mean())
        self.data["Crop Residues"] = self.data["Crop Residues"].fillna(self.data["Crop Residues"].mean())
        self.data["Net Forest conversion"] = self.data["Net Forest conversion"].fillna(self.data["Net Forest conversion"].mean())
        self.data["Food Household Consumption"] = self.data["Food Household Consumption"].fillna(self.data["Food Household Consumption"].mean())
        self.data["Forestland"] = self.data["Forestland"].fillna(self.data["Forestland"].mean())
        self.data["IPPU"] = self.data["IPPU"].fillna(self.data["IPPU"].mean())
        self.data["Manure applied to Soils"] = self.data["Manure applied to Soils"].fillna(self.data["Manure applied to Soils"].mean())
        self.data["Manure Management"] = self.data["Manure Management"].fillna(self.data["Manure Management"].mean())
        self.data["Fires in humid tropical forests"] = self.data["Fires in humid tropical forests"].fillna(self.data["Fires in humid tropical forests"].mean())
        self.data["On-farm energy use"] = self.data["On-farm energy use"].fillna(self.data["On-farm energy use"].mean())

    def calc_corr(self):
        le =LabelEncoder()
        self.data["Area"] = le.fit_transform(self.data["Area"])
        corr =self.data.corr()
        return corr

    def run_ep_print(self):
        self.print_data_col()
        self.null_data()
        self.head()
        print(self.calc_corr()["total_emission"].sort_values(ascending=True))

    def return_data(self):
        return self.data

    def run_ep_function(self):
        self.fill_null_data()
        return self.return_data()





