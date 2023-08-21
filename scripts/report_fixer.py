from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re


def extract_timestamp(path: Path):
    """ example report_20230820-214937.txt return 20230820-214937 as a string"""
    return path.stem.split("_")[1]


OUTPUT_DIR = Path("output/reports/face-recognition/summary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
input_path = Path(
    "output/reports/face-recognition/classification_reports/report_20230820-214937.txt")

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
    you = re.search(r"You\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", section)
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

# Create a pivot table for the heatmap
pivot_table = df.pivot_table(index="Target Percentage", columns=[
                             "SMOTE", "Face Detection"], values="Weighted Avg F1")


# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", linewidths=0.5,
            cbar_kws={"label": "Weighted Avg F1 Score"})

plt.title("Performance Comparison based on Weighted Avg F1 Score")
ylabel = "Percentage of Target Face in Data Set (%)"
plt.ylabel(ylabel)
plt.xlabel("Combination of SMOTE and Face Detection")
# plt.show()
plt.savefig(OUTPUT_DIR/f"heatmap_F1_{extract_timestamp(input_path)}.png")

# Assuming you have already defined 'df' from previous steps

# Create a pivot table for the heatmap
pivot_table_recall = df.pivot_table(index="Target Percentage", columns=[
                                    "SMOTE", "Face Detection"], values="You Recall")

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table_recall, annot=True, cmap="YlGnBu",
            linewidths=0.5, cbar_kws={"label": "Recall Score"})

plt.title("Performance Comparison based on Recall")
plt.ylabel(ylabel)
plt.xlabel("Combination of SMOTE and Face Detection")
# plt.show()
plt.savefig(OUTPUT_DIR/f"heatmap_recall_{extract_timestamp(input_path)}.png")
