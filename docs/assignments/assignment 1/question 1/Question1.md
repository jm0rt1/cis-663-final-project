# Question 1

## Introduction
Finding model accuracy 


## Code
```python
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Assuming the data is in this format
# Load the data from the text file
df = pd.read_csv(
    'docs/assignments/assignment 1/question 1/s048r_202307.txt', sep='\t')


# Binarize the labels
df['test.subject'] = df['test.subject'].apply(
    lambda x: 1 if x == 's048' else 0)
df['test.out'] = df['test.out'].apply(lambda x: 1 if x == 's048' else 0)

# Get the labels and predictions
y_true = df['test.subject']
y_pred = df['test.out']

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Print confusion matrix
print(f"Confusion Matrix:\nTP={tp} FN={fn}\nFP={fp} TN={tn}")

# Compute Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

# Compute FMR
fmr = fp / (fp + tn)
print(f"FMR: {fmr}")

# Compute FNMR
fnmr = fn / (fn + tp)
print(f"FNMR: {fnmr}")

# Compute Precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")

# Compute Recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall}")


with open('docs/assignments/assignment 1/question 1/results.txt', 'w') as f:
    f.write(f"Confusion Matrix:\nTP={tp} FN={fn}\nFP={fp} TN={tn}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"FMR: {fmr}\n")
    f.write(f"FNMR: {fnmr}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
```
## Output
    
        Confusion Matrix:
        TP=156 FN=48
        FP=6 TN=194
        Accuracy: 0.8663366336633663
        FMR: 0.03
        FNMR: 0.23529411764705882
        Precision: 0.9629629629629629
        Recall: 0.7647058823529411
