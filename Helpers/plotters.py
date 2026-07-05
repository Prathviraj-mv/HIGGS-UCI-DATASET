import seaborn as sns
import matplotlib.pyplot as plt
from Definitions.constants import OUTPUT_DIR_p,OUTPUT_DIR_rf

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
        plt.savefig(OUTPUT_DIR_p/"heatmap.png")


    def  plot_confusion_matrix(self,value):
        plt.figure(figsize=(20,15))
        sns.heatmap()
        sns.heatmap(value,
                    fmt=".1f",
                    annot=True,
                    vmin=-1,
                    vmax=1,
                    linewidths=0.5,
                    )
        plt.title("CONFUSION Matrix ")
        plt.savefig(OUTPUT_DIR_rf / "heatmap.png")


if __name__ == "__main__":
        plot =PLOT()
        plot.plot_correlation()