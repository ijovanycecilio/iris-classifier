# src/train.py
import argparse
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


def main(test_size: float = 0.2, random_state: int = 42) -> None:
    # 1️⃣ Load the data
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2️⃣ Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3️⃣ Train a Decision Tree
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    # 4️⃣ Make predictions
    y_pred = clf.predict(X_test)

    # 5️⃣ Evaluate performance
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")

    cm = confusion_matrix(y_test, y_pred)

    # 6️⃣ Save confusion-matrix figure
    os.makedirs("outputs", exist_ok=True)  # create folder if missing
    fig_path = os.path.join("outputs", "confusion_matrix.png")

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Decision Tree)")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    print(f"Saved confusion-matrix figure to: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree on the Iris dataset and save a confusion matrix."
    )
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data used for testing (default 0.2).")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    main(test_size=args.test_size, random_state=args.random_state)
