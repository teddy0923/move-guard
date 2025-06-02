#!/usr/bin/env python3
import pandas as pd
import re
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# 1. Load data (single file with ID, label, and features)
df = pd.read_csv("data/labeled_features/squat_features_front_labeled.csv")

# 2. Auto-select feature columns
feature_cols = [
    col for col in df.columns
    if re.search(r"(angle|flexion).*(min|max|range|mean)", col, flags=re.IGNORECASE)
]
X = df[feature_cols]
y = df["label"]

# 3. Define classifier
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight={1: 2.0, 2: 1.0, 3: 0.5},
    min_samples_split=2,
    min_samples_leaf=1
)

# 4. 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
scoring = ["accuracy", "f1_macro", "f1_weighted"]
scores = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=False)

print("Cross-validation results (mean ± std):")
for metric in scoring:
    arr = scores[f"test_{metric}"]
    print(f"  {metric}: {arr.mean():.3f} ± {arr.std():.3f}")
print()

# 5. Final hold-out split for plots, error CSV, and model pickle
df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)
X_train, y_train = df_train[feature_cols], df_train["label"]
X_test,  y_test  = df_test[feature_cols],  df_test["label"]

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 6. Classification report
print("Hold-out classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Confusion matrix → save, show, close
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.savefig("confusion_matrix.png")
plt.show()
plt.close()

# 8. Feature importances → CSV + bar chart → save, show, close
importances = clf.feature_importances_
fi = pd.DataFrame({"feature": feature_cols, "importance": importances})
fi = fi.sort_values("importance", ascending=False)
fi.to_csv("feature_importances.csv", index=False)

ax = fi.plot.bar(x="feature", y="importance", legend=False)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()
plt.close()

# 9. Serialize model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)

# 10. Save predictions + errors with ID for review
results = df_test[["ID"]].copy()
results["true_label"]      = y_test.values
results["predicted_label"] = y_pred
results["correct"]         = results["true_label"] == results["predicted_label"]

results.to_csv("predictions_with_ID.csv", index=False)
results[~results["correct"]].to_csv("errors_to_review.csv", index=False)
