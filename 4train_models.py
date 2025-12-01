import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

def train_models(data_dir: str):
    print("\n📂 Loading vectorized data...")
    X_train = joblib.load(f"{data_dir}/X_train.pkl")
    X_test = joblib.load(f"{data_dir}/X_test.pkl")
    y_train = joblib.load(f"{data_dir}/y_train.pkl")
    y_test = joblib.load(f"{data_dir}/y_test.pkl")

    print("\n🚀 Training Models...\n")

    # ---------------------------------------------------------
    # LOGISTIC REGRESSION
    # ---------------------------------------------------------
    print("🔵 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=500, solver="liblinear")
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    joblib.dump(lr, f"{data_dir}/logistic_regression.pkl")
    print(f"✔ Logistic Regression Accuracy: {lr_acc:.4f}")

    # ---------------------------------------------------------
    # SVM
    # ---------------------------------------------------------
    print("\n🟣 Training Linear SVM...")
    svm = LinearSVC(dual=False)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    joblib.dump(svm, f"{data_dir}/svm_model.pkl")
    print(f"✔ SVM Accuracy: {svm_acc:.4f}")

    # ---------------------------------------------------------
    # NAIVE BAYES
    # ---------------------------------------------------------
    print("\n🟡 Training Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    joblib.dump(nb, f"{data_dir}/naive_bayes.pkl")
    print(f"✔ Naive Bayes Accuracy: {nb_acc:.4f}")

    # ---------------------------------------------------------
    # DETAILED REPORTS
    # ---------------------------------------------------------
    print("\n📊 MODEL PERFORMANCE REPORTS\n")
    print("LOGISTIC REGRESSION REPORT:\n", classification_report(y_test, lr_pred))
    print("SVM REPORT:\n", classification_report(y_test, svm_pred))
    print("NAIVE BAYES REPORT:\n", classification_report(y_test, nb_pred))

    # ---------------------------------------------------------
    # CONFUSION MATRICES (optional)
    # ---------------------------------------------------------
    print("CONFUSION MATRIX — LOGISTIC REGRESSION:\n", confusion_matrix(y_test, lr_pred))
    print("CONFUSION MATRIX — SVM:\n", confusion_matrix(y_test, svm_pred))
    print("CONFUSION MATRIX — NAIVE BAYES:\n", confusion_matrix(y_test, nb_pred))

    print("\n🎉 Training complete! Best model likely = SVM or Logistic Regression")


if __name__ == "__main__":
    train_models(r"C:\\Users\\vansh\\OneDrive\\Desktop\\new_classification\\data")
