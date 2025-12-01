import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

print("\n📌 Loading model and test data...")

# -------- Absolute paths --------
model_path = r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\logistic_regression.pkl"
X_test_path = r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\X_test.pkl"
y_test_path = r"C:\Users\vansh\OneDrive\Desktop\new_classification\data\y_test.pkl"

# -------- Load model + test data --------
model = joblib.load(model_path)
X_test = joblib.load(X_test_path)
y_test = joblib.load(y_test_path)

print("\n🔍 Making predictions...")
y_pred = model.predict(X_test)

# -------------------------------------------------------------
# Metrics
# -------------------------------------------------------------
print("\n📊 PERFORMANCE METRICS")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f" Accuracy  : {accuracy:.4f}")
print(f" Precision : {precision:.4f}")
print(f" Recall    : {recall:.4f}")
print(f" F1-score  : {f1:.4f}")

# Classification report
print("\n🔎 CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

# -------------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------------
print("\n📌 CONFUSION MATRIX")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["Pred Fake", "Pred Real"],
            yticklabels=["Actual Fake", "Actual Real"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# AUC-ROC Score
# -------------------------------------------------------------
print("\n💠 Calculating AUC-ROC...")

try:
    scores = model.decision_function(X_test)
except:
    scores = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, scores)
print(f" AUC-ROC: {auc:.4f}")

# -------- ROC Curve Plot --------
fpr, tpr, _ = roc_curve(y_test, scores)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
