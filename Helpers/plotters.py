import seaborn as sns
import matplotlib.pyplot as plt

from Helpers.data_extract_and_preprocessing import DATA

class PLOT:
    def __init__(self):
        pass

    def plot_correlation(self):
        correlation = DATA().calc_corr()

        plt.figure(figsize=(10,5))
        sns.heatmap(correlation,
            fmt=".1f",
            annot=True,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            )
        plt.title("Correlation Matrix Heatmap")
        plt.show()

