# Practical 6: Classification Techniques

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt


df = pd.read_csv("datasets/data.csv")

df["Calories"] = df["Calories"].fillna(df["Calories"].mean())

df["Target"] = (df["Calories"] > df["Calories"].median()).astype(int)

X = df[["Duration", "Pulse", "Maxpulse"]]
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


def evaluate_model(name, model):
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Precision:",
          precision_score(y_test, y_pred))

    print("Recall:",
          recall_score(y_test, y_pred))

    print("F1 Score:",
          f1_score(y_test, y_pred))

    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr,
             label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.show()


evaluate_model("Logistic Regression",LogisticRegression())
evaluate_model("kNN",KNeighborsClassifier(n_neighbors=5))
evaluate_model("Decision Tree",DecisionTreeClassifier(random_state=42))
evaluate_model("SVM",SVC(probability=True))
print("\nClassification Completed Successfully!")