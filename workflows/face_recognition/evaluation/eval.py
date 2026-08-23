import os
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

MODEL_PATH = "/home/debashis/SIH'26/Inter Clg/Attendance-System/models/knn_model.pkl"
TEST_DIR = "/home/debashis/SIH'26/Inter Clg/Attendance-System/data/test_data_embeddings"

knn = joblib.load(MODEL_PATH)

print("KNN model loaded successfully.")

X_test = []
y_test = []

for filename in sorted(os.listdir(TEST_DIR)):

    if not filename.endswith(".npy"):
        continue

    # Roll number
    roll = os.path.splitext(filename)[0]

    file_path = os.path.join(
        TEST_DIR,
        filename
    )

    embeddings = np.load(file_path)

    print(
        f"Roll {roll}: {embeddings.shape}"
    )

    for embedding in embeddings:

        X_test.append(embedding)
        y_test.append(roll)


# Convert to NumPy
X_test = np.array(
    X_test,
    dtype=np.float32
)


y_test = np.array(y_test)

print("\nTest Dataset:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("RESULT")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nIndividual Predictions:")

for actual, predicted in zip(
    y_test,
    y_pred
):

    status = "✓" if actual == predicted else "✗"

    print(
        f"{status} "
        f"Actual: {actual} | "
        f"Predicted: {predicted}"
    )
    
labels = sorted(
    np.unique(y_test)
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot()

plt.title(
    "Face Recognition - KNN"
)

plt.show()