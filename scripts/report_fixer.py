# Redefining the refactored functions and main function after importing the required libraries
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re


def extract_timestamp(path: Path) -> str:
    """ Extract timestamp from given file path. """
    return path.stem.split("_")[1]


def create_output_directory(path: Path) -> Path:
    timestamp = extract_timestamp(path)
    OUTPUT_DIR = Path(
        f"output/reports/face-recognition/summary/summary_for_{timestamp}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def extract_report_data(path: Path) -> pd.DataFrame:
    with open(path.as_posix(), 'r') as f:
        report = f.read()

    # Removing the first split result which is empty
    sections = report.split("Classification Report")[1:]
    data = []

    for section in sections:
        target_percentage = int(
            re.search(r"Percentage of Target in Dataset: (\d+)", section).group(1))
        smote = "SMOTE Resampled = True" in section
        face_detection = "Face Detection Used = True" in section
        not_you = re.search(
            r"Not You\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", section)
        you = re.search(r"You\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", section)
        macro_avg = re.search(
            r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", section)
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
    return df


def generate_visualizations(df: pd.DataFrame, output_dir: Path):
    # Line plot
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x='Target Percentage', y='Weighted Avg F1',
                 hue='SMOTE', style='Face Detection', markers=True, dashes=False)
    plt.title('Weighted Avg F1 Score vs. Target Percentage')
    plt.xlabel('Target Percentage (%)')
    plt.ylabel('Weighted Avg F1 Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        output_dir / f"F1_vs_Target_Percentage_{extract_timestamp(input_path)}.png")
    plt.close()

    # Heatmap for correlation
    corr = df.drop(columns=['SMOTE', 'Face Detection',
                   'Target Percentage']).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title('Correlation Heatmap of Metrics')
    plt.tight_layout()
    plt.savefig(
        output_dir / f"heatmap_correlation_{extract_timestamp(input_path)}.png")
    plt.close()

    # Pairplot
    selected_columns = ['Not You Precision', 'Not You Recall', 'Not You F1',
                        'You Precision', 'You Recall', 'You F1',
                        'Macro Avg F1', 'Weighted Avg F1']
    pairplot = sns.pairplot(df[selected_columns + ['SMOTE', 'Face Detection']],
                            hue='SMOTE & Face Detection', corner=True, diag_kind='kde', plot_kws={'alpha': 0.6})
    pairplot.fig.suptitle(
        "Pairwise Relationships and Distributions of Key Metrics", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f"pairplot_{extract_timestamp(input_path)}.png")
    plt.close()


def main(input_path: Path):
    output_dir = create_output_directory(input_path)
    df = extract_report_data(input_path)
    df['SMOTE & Face Detection'] = df['SMOTE'].astype(
        str) + ' & ' + df['Face Detection'].astype(str)
    generate_visualizations(df, output_dir)


input_path = Path(
    "output/reports/face-recognition/classification_reports/report_20230820-214937.txt")

# Re-running the main function after redefining the refactored functions
main(input_path)

# Return message for completion
"Refactored code executed successfully!"
