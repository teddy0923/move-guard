#!/usr/bin/env python3
import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# 1. Load data (single file with ID, label, and features)
df = pd.read_csv("data/labeled_features/squat_features_labeled.csv")

# ---- handle missing labels ----
n_missing = df["label"].isna().sum()
if n_missing > 0:
    print(f"Dropping {n_missing} rows with missing labels")
    df = df.dropna(subset=["label"])
# --------------------------------


# 2. Auto-select feature columns
feature_cols = [
    c for c in df.columns
    if re.search(r"(angle|flexion|alignment|ratio|distance).*(min|max|range|mean)", c)
]


# 3. Stratified train/test split on the entire DataFrame
df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

X_train = df_train[feature_cols]
y_train = df_train["label"]
X_test  = df_test[feature_cols]
y_test  = df_test["label"]

# 4. Train weighted Random Forest
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    min_samples_split=2,  # <-- new parameter
    min_samples_leaf=1  # <-- new parameter
    #,
    #class_weight={1: 2.0, 2: 1.0, 3: 0.5}
)
clf.fit(X_train, y_train)

# 5. Evaluate & print report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# 6. Confusion matrix plot → save
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.savefig("confusion_matrix.png")
plt.show ()
plt.close()

# 7. Feature importances → CSV + bar chart
importances = clf.feature_importances_
fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=True)
fi.to_csv("feature_importances.csv", index=False)

# Horizontal bar chart
fi.plot.barh(x="feature", y="importance", legend=True)
plt.xlabel("importance")
plt.ylabel("feature")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()
plt.close()

# 8. Serialize model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)

# 9. Save mis-classified IDs for review
results = df_test[["ID"]].copy()
results["true_label"]      = y_test.values
results["predicted_label"] = y_pred
results["correct"]         = results["true_label"] == results["predicted_label"]

# All predictions, if you want
results.to_csv("predictions_with_ID.csv", index=False)
# Just the errors
errors = results[~results["correct"]]
errors.to_csv("errors_to_review.csv", index=False)
