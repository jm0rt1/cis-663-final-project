import enum
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re

input_path = Path(
    "output/reports/face-recognition/classification_reports/report_20230831-175957.txt")


class Columns(str, enum.Enum):
    """Columns in the report"""
    TARGET_PERCENTAGE = "Target Percentage"
    SMOTE = "SMOTE"
    FACE_DETECTION = "Face Detection"
    NOT_YOU_PRECISION = "Not You Precision"
    NOT_YOU_RECALL = "Not You Recall"
    NOT_YOU_F1 = "Not You F1"
    YOU_PRECISION = "You Precision"
    YOU_RECALL = "You Recall"
    YOU_F1 = "You F1"
    MACRO_AVG_F1 = "Macro Avg F1"
    WEIGHTED_AVG_F1 = "Weighted Avg F1"


def extract_timestamp(path: Path):
    """ Example report_20230820-214937.txt return 20230820-214937 as a string"""
    return path.stem.split("_")[1]


OUTPUT_DIR = Path(
    f"output/reports/face-recognition/summary/summary_for_{extract_timestamp(input_path)}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with open(input_path.as_posix(), 'r') as f:
    report = f.read()

# Extracting information from report
sections = report.split("Classification Report")
sections = sections[1:]  # Removing the first split result which is empty

data = []

for section in sections:
    target_percentage = int(
        re.search(r"Percentage of Target in Dataset: (\d+)", section).group(1))
    smote = True if "SMOTE Resampled = True" in section else False
    face_detection = True if "Face Detection Used = True" in section else False

    not_you = re.search(r"Not You\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", section)
    you = re.search(
        r"^\s*You\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", section, re.MULTILINE)

    macro_avg = re.search(r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", section)
    weighted_avg = re.search(
        r"weighted avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", section)

    data.append([
        target_percentage, smote, face_detection,
        float(not_you.group(1)), float(
            not_you.group(2)), float(not_you.group(3)),
        float(you.group(1)), float(you.group(2)), float(you.group(3)),
        float(macro_avg.group(1)), float(weighted_avg.group(1))
    ])

df = pd.DataFrame(data, columns=[
    "Target Percentage", "SMOTE", "Face Detection",
    "Not You Precision", "Not You Recall", "Not You F1",
    "You Precision", "You Recall", "You F1",
    "Macro Avg F1", "Weighted Avg F1"
])

# Sorting by Weighted Avg F1 to get the top combination
best_combination = df.sort_values(
    by="Weighted Avg F1", ascending=False).iloc[0]

with open(OUTPUT_DIR/f"report_{extract_timestamp(input_path)}.csv", 'w') as f:
    f.write(df.to_csv(index=False))
    f.write(f"\n\n\nBest Combination based on Weighted Avg F1:\n")
    f.write(f"Target Percentage: {best_combination['Target Percentage']}\n")
    f.write(f"SMOTE: {best_combination['SMOTE']}\n")
    f.write(f"Face Detection: {best_combination['Face Detection']}\n")
    f.write(f"Weighted Avg F1 Score: {best_combination['Weighted Avg F1']}\n")


# Assuming you have already defined 'df' from previous steps


# Assuming you have already defined 'df' from previous steps


# Create a pivot table for the heatmap


def plot_heatmap(input_path: Path, df: pd.DataFrame, target_metric: Columns):
    ylabel = "Percentage of Target Face in Data Set (%)"

    pivot_table_recall = df.pivot_table(index="Target Percentage", columns=[
        "SMOTE", "Face Detection"], values=target_metric.value)
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table_recall, annot=True, cmap="YlGnBu",
                linewidths=0.5, cbar_kws={"label": target_metric.value})  # type: ignore

    plt.title(f"Performance Comparison based on {target_metric.value}")
    plt.ylabel(ylabel)
    plt.xlabel("Combination of SMOTE and Face Detection")

    plt.savefig(
        OUTPUT_DIR/f"heatmap_{target_metric.lower().replace(' ', '_')}_{extract_timestamp(input_path)}.png")


# Precisions
plot_heatmap(input_path, df, Columns.NOT_YOU_PRECISION)
plot_heatmap(input_path, df, Columns.YOU_PRECISION)

# Recalls
plot_heatmap(input_path, df, Columns.NOT_YOU_RECALL)
plot_heatmap(input_path, df, Columns.YOU_RECALL)

# F1 Scores
plot_heatmap(input_path, df, Columns.YOU_F1)
plot_heatmap(input_path, df, Columns.NOT_YOU_F1)

# Average F1 Scores
plot_heatmap(input_path, df, Columns.MACRO_AVG_F1)
plot_heatmap(input_path, df, Columns.WEIGHTED_AVG_F1)


# Plot line plot
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x="Target Percentage", y='Weighted Avg F1',
             hue='SMOTE', style='Face Detection', markers=True, dashes=False)
plt.title('Weighted Avg F1 Score vs. Target Percentage')
plt.xlabel('Target Percentage (%)')
plt.ylabel('Weighted Avg F1 Score')
plt.grid(True)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR/f"F1_vs_Target_Percentage{extract_timestamp(input_path)}.png")


# Visualization 5: Heatmap for correlation between metrics

# Compute the correlation matrix
corr = df.drop(columns=['SMOTE', 'Face Detection',
                        'Target Percentage']).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Heatmap of Metrics')
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR/f"heatmap_correlation_{extract_timestamp(input_path)}.png")


# Visualization: Pairplot to show relationships and distributions between key metrics

# Selecting key metrics for the pairplot
selected_columns = ['Not You Precision', 'Not You Recall', 'Not You F1',
                    'You Precision', 'You Recall', 'You F1',
                    'Macro Avg F1', 'Weighted Avg F1']


"""
Key Features of the Pairplot:
Scatter Plots: The off-diagonal plots are scatter plots.
They show the relationship between two variables. For instance,
the scatter plot at the intersection of "Not You Precision" (row) and
"You Precision" (column) visualizes the relationship between these two metrics.
Distribution Plots: The diagonal plots are distribution plots (kernel density estimates)
for each metric. They give an idea of how the values for each metric are distributed.
Visualization of Correlations: Scatter plots in the pairplot provide a visual representation
of the correlation between pairs of metrics. The tighter and more linear the clustering of points, 
the stronger the correlation.
"""
# Creating the pairplot
pairplot = sns.pairplot(
    df[selected_columns], corner=True, diag_kind='kde', plot_kws={'alpha': 0.6})
pairplot.fig.suptitle(
    "Pairwise Relationships and Distributions of Key Metrics", y=1.02)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR/f"pairplot_{extract_timestamp(input_path)}.png")
