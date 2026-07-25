from Definitions.constants import OUTPUT_DIR_p
from Helpers.data_extract_and_preprocessing import DATA
from Helpers.plotters import PLOT


class EDA:
    def __init__(self):
        self.data = DATA().return_data()

    def run(self):
        plt = PLOT()
        
        # Original plots
        plt.plot_correlation(OUTPUT_DIR_p)
        plt.plot_joint_plot(path=OUTPUT_DIR_p, data=self.data, x="jet1_pt", y="label")

        plt.plot_scatter_plot(path=OUTPUT_DIR_p, data=self.data, x="jet1_pt", y="label")
        plt.plot_count_plot(path=OUTPUT_DIR_p, data=self.data)

        plt.histplot(path=OUTPUT_DIR_p, data=self.data, x="jet1_pt")
        plt.kdeplot(path=OUTPUT_DIR_p, data=self.data, x="jet1_pt")
        plt.pairplot(path=OUTPUT_DIR_p, data=self.data, list_=["jet1_pt", "jet2_pt", "jet3_pt", "jet4_pt", "label"])
        
        # New enhanced plots
        plt.plot_correlation_with_target(OUTPUT_DIR_p, self.data)
        
        # Boxplots for key features
        key_features = ["jet1_pt", "jet2_pt", "m_jj", "m_bb", "m_wwbb"]
        for feature in key_features:
            plt.plot_boxplot(OUTPUT_DIR_p, self.data, feature)
            plt.plot_violin(OUTPUT_DIR_p, self.data, feature)
            plt.plot_stacked_histogram(OUTPUT_DIR_p, self.data, feature)
        
        # 3D scatter plot for top 3 features
        plt.plot_3d_scatter(OUTPUT_DIR_p, self.data, "jet1_pt", "jet2_pt", "m_jj")
        
        # Radar plot for feature comparison
        radar_features = ["jet1_pt", "jet2_pt", "m_jj", "m_bb", "m_wwbb", "lepton_pt"]
        plt.plot_radar(OUTPUT_DIR_p, self.data, radar_features)
        
        # Distribution overview for all key features
        plt.plot_distribution_all(OUTPUT_DIR_p, self.data, key_features)


if __name__ == "__main__":
    plot = EDA()
    plot.run()
