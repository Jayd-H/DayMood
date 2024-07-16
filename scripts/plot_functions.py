import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import seaborn as sns


def plot_feature_importance(feature_importance_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        feature_importance_df["Feature"],
        feature_importance_df["Importance"],
        color="skyblue",
    )
    ax.set_xlabel("Importance (%)")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance")
    ax.invert_yaxis()

    fig.text(0.5, -0.1, wrap=True, horizontalalignment="center", fontsize=12)

    fig.tight_layout()
    return fig


def plot_category_over_time(data_cleaned, category):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        data_cleaned["Date"],
        data_cleaned[category],
        label=f"{category} (Raw Data)",
        alpha=0.7,
    )

    y_smooth = gaussian_filter1d(data_cleaned[category], sigma=3)
    ax.plot(
        data_cleaned["Date"],
        y_smooth,
        linestyle="dashed",
        alpha=0.7,
        label=f"{category} (Best Fit)",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.set_title(f"{category} Over Time")
    ax.legend()

    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))

    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig


def plot_all_categories_over_time(data_cleaned, categories, features_to_exclude):
    fig, ax = plt.subplots(figsize=(14, 8))

    for category in categories:
        if category not in features_to_exclude:
            y_smooth = gaussian_filter1d(data_cleaned[category], sigma=3)
            ax.plot(
                data_cleaned["Date"],
                y_smooth,
                linestyle="dashed",
                alpha=0.7,
                label=f"{category} (Best Fit)",
            )

    ax.set_xlabel("Date")
    ax.set_ylabel("Score")
    ax.set_title("All Categories Over Time with Best Fit Lines")
    ax.legend()

    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))

    plt.xticks(rotation=45)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(data_cleaned):
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = data_cleaned.drop(columns=["Day", "Date"]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_box_plots(data_cleaned):
    fig, ax = plt.subplots(figsize=(14, 8))
    data_to_plot = data_cleaned.drop(columns=["Day", "Date"])
    sns.boxplot(data=data_to_plot, orient="h", ax=ax)
    ax.set_title("Box Plots for Each Category")
    fig.tight_layout()
    return fig
