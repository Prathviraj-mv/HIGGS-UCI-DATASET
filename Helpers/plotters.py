import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from Helpers.data_extract_and_preprocessing import DATA

class PLOT:
    def __init__(self):
        pass

    def plot_correlation(self,x):
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
        plt.savefig(x /"heatmap.jpg", format='jpg', dpi=150)
        plt.close()



    def  plot_confusion_matrix(self,value,x):
        plt.figure(figsize=(20,15))
        sns.heatmap(value,
                    fmt=".1f",
                    annot=True,
                    vmin=-1,
                    vmax=1,
                    linewidths=0.5,
                    )
        plt.title("CONFUSION Matrix ")
        plt.savefig(x / "confusion.jpg", format='jpg', dpi=150)
        plt.close()


    def plot_joint_plot(self,path,data,x,y):
        plt.figure(figsize=(20,15))
        sns.jointplot(data=data,x=x,y=y)
        plt.title("Joint Plot ")
        plt.savefig(path / f"{x}_{y}_joint_plot.jpg", format='jpg', dpi=150)
        plt.close()



    def plot_scatter_plot(self,path,data,x,y):
        plt.figure(figsize=(20,15))
        sns.scatterplot(data=data,x=x,y=y)
        plt.title("scatterplot")
        plt.savefig(path / f"{x}_{y}scatter_plot.jpg", format='jpg', dpi=150)
        plt.close()

    def plot_count_plot(self,path,data):
        plt.figure(figsize=(20, 15))
        sns.countplot(data=data, x="label")
        plt.title("countplot")
        plt.savefig(path / f"countplot.jpg", format='jpg', dpi=150)
        plt.close()

    def histplot(self,path,data,x):
        plt.figure(figsize=(20, 15))
        sns.histplot(
            data=data,
            x=x,
            hue="label",
            bins=50,
            kde=True,
            stat="density",
            common_norm=False
        )
        plt.title("histplot")
        plt.savefig(path / f"histplot.jpg", format='jpg', dpi=150)
        plt.close()

    def kdeplot(self,path,data,x):
        plt.figure(figsize=(20, 15))
        sns.kdeplot(
            data=data,
            x=x,
            hue="label",
            fill=True
        )
        plt.title("kdeplot")
        plt.savefig(path / f"kdeplot.jpg", format='jpg', dpi=150)
        plt.close()

    def pairplot(self,path,data,list_):
        # Sample data to avoid memory issues
        sample_data = data[list_].sample(n=min(1000, len(data)), random_state=42)
        plt.figure(figsize=(20, 15))
        sns.pairplot(
            sample_data,
            hue="label",
            diag_kind="hist",  # Use histogram instead of KDE to avoid computational issues
            plot_kws={'alpha': 0.6}
        )
        plt.suptitle("Pairplot (Sampled Data)", y=1.02)
        plt.savefig(path / f"pairplot.jpg", format='jpg', dpi=150)
        plt.close()

    def plot_boxplot(self, path, data, feature):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data,
                    x="label", 
                    y=feature)

        plt.title(f"Boxplot of {feature} by Label")
        plt.savefig(path / f"boxplot_{feature}.jpg", format='jpg', dpi=150)
        plt.close()

    def plot_violin(self, path, data, feature):
        # Sample data to avoid memory issues
        sample_data = data.sample(n=min(5000, len(data)), random_state=42)
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=sample_data, x="label", y=feature)
        plt.title(f"Violin Plot of {feature} by Label")
        plt.savefig(path / f"violin_{feature}.jpg", format='jpg', dpi=150)
        plt.close()

    def plot_feature_importance(self, path, importance_dict):
        plt.figure(figsize=(12, 8))
        features = list(importance_dict.keys())
        importance = list(importance_dict.values())
        sns.barplot(x=importance, y=features)
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.savefig(path / "feature_importance.jpg", format='jpg', dpi=150)
        plt.close()

    def plot_distribution_all(self, path, data, features):
        fig, axes = plt.subplots(len(features), 2, figsize=(15, 4*len(features)))
        for i, feature in enumerate(features):
            sns.histplot(data=data, x=feature, hue="label", ax=axes[i, 0], kde=True)
            axes[i, 0].set_title(f"Distribution of {feature}")
            sns.boxplot(data=data, x="label", y=feature, ax=axes[i, 1])
            axes[i, 1].set_title(f"Boxplot of {feature}")
        plt.tight_layout()
        plt.savefig(path / "distribution_all_features.jpg", format='jpg', dpi=150)
        plt.close()

    def plot_correlation_with_target(self, path, data):
        corr_with_target = data.corr()["label"].drop("label").sort_values(ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=corr_with_target.values, y=corr_with_target.index)
        plt.title("Feature Correlation with Target")
        plt.xlabel("Correlation Coefficient")
        plt.ylabel("Features")
        plt.savefig(path / "correlation_with_target.jpg", format='jpg', dpi=150)
        plt.close()

    def plot_3d_scatter(self, path, data, x, y, z):
        # Sample data to avoid memory issues
        sample_data = data.sample(n=min(2000, len(data)), random_state=42)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(sample_data[x], sample_data[y], sample_data[z], c=sample_data["label"], cmap='viridis', alpha=0.6)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        plt.title(f"3D Scatter Plot: {x}, {y}, {z}")
        plt.colorbar(scatter)
        plt.savefig(path / f"3d_scatter_{x}_{y}_{z}.jpg", format='jpg', dpi=150)
        plt.close()

    def plot_stacked_histogram(self, path, data, feature):
        plt.figure(figsize=(12, 6))
        signal = data[data["label"] == 1][feature]
        background = data[data["label"] == 0][feature]
        plt.hist([signal, background], bins=50, stacked=True, label=["Signal", "Background"], alpha=0.7)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.title(f"Stacked Histogram of {feature}")
        plt.legend()
        plt.savefig(path / f"stacked_hist_{feature}.jpg", format='jpg', dpi=150)
        plt.close()

    def plot_radar(self, path, data, features):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        signal_means = data[data["label"] == 1][features].mean()
        background_means = data[data["label"] == 0][features].mean()
        
        angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
        signal_means = np.concatenate((signal_means.values, [signal_means.values[0]]))
        background_means = np.concatenate((background_means.values, [background_means.values[0]]))
        angles += angles[:1]
        
        ax.plot(angles, signal_means, 'o-', linewidth=2, label="Signal")
        ax.plot(angles, background_means, 'o-', linewidth=2, label="Background")
        ax.fill(angles, signal_means, alpha=0.25)
        ax.fill(angles, background_means, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)
        plt.title("Radar Plot of Feature Means")
        plt.legend()
        plt.savefig(path / "radar_plot.jpg", format='jpg', dpi=150)
        plt.close()


