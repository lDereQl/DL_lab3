import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# === Configuration ===
BASE_PATH = './filtered_coco_yolo'  # Update this if your path is different
LABEL_PATHS = {
    'train': os.path.join(BASE_PATH, 'labels/train'),
    'val': os.path.join(BASE_PATH, 'labels/val')
}

# Class mappings
CLASS_MAPPING = {0: 'boat', 1: 'bird'}


# === Functions ===
def collect_data(label_dir):
    """
    Collect class data from YOLO-formatted label files.
    """
    class_counts = Counter()
    if not os.path.exists(label_dir):
        print(f"Path not found: {label_dir}")
        return class_counts

    for file_name in os.listdir(label_dir):
        with open(os.path.join(label_dir, file_name), 'r') as file:
            for line in file:
                class_id = int(line.split()[0])
                class_counts[CLASS_MAPPING[class_id]] += 1
    return class_counts


def plot_class_distribution(train_counts, val_counts):
    """
    Plot class distributions for training and validation datasets.
    """
    data = {
        'Class': ['boat', 'bird'],
        'Train Count': [train_counts['boat'], train_counts['bird']],
        'Validation Count': [val_counts['boat'], val_counts['bird']]
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Class', y='Train Count', data=df, color='lightblue', label='Train Set')
    sns.barplot(x='Class', y='Validation Count', data=df, color='salmon', label='Validation Set')
    plt.title('Class Distribution in Train and Validation Sets')
    plt.legend()
    plt.show()
    print(df)


# === Main Execution ===
if __name__ == "__main__":
    print("Analyzing class distribution...")

    # Collect data
    train_counts = collect_data(LABEL_PATHS['train'])
    val_counts = collect_data(LABEL_PATHS['val'])

    # Plot and display
    plot_class_distribution(train_counts, val_counts)
