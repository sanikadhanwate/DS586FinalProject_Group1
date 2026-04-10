# step 1: Run preprocessing first to generate the two output CSVs
# Then load them:
import pandas as pd
import numpy as np

meta_ml       = pd.read_csv("../meta_dataset_ml_ready.csv", index_col="PATIENT")
meta_readable = pd.read_csv("../meta_dataset_readable.csv", index_col="PATIENT")

#Step 2: Define X and y
# X = symptom binary features + demographic dummies + normalized age/num_symptoms
drop_cols = ["PATHOLOGY", "SYMPTOM_LIST", "CAREPLAN_LIST", 
             "AGE_BEGIN", "AGE_END",  # already normalized versions exist
             "NUM_CAREPLANS"]

X = meta_ml.drop(columns=[c for c in drop_cols if c in meta_ml.columns])
y = meta_readable["PATHOLOGY"]   # target: disease name

print("X shape:", X.shape)
print("Unique classes:", y.nunique())

# Filter to pathologies with enough samples
from collections import Counter
class_counts = Counter(y)
valid_classes = {cls for cls, cnt in class_counts.items() if cnt >= 3}
mask = y.isin(valid_classes)
X = X[mask]
y = y[mask]
print(f"After filtering: {len(X)} patients, {y.nunique()} classes")

#step 3:train/test & split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,         # 80/20 split
    random_state=42,
    stratify=y             # keeps class distribution balanced across splits
)
print("Train size:", X_train.shape, "| Test size:", X_test.shape)

#step 4: Train SVM
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Encode string labels to integers (SVM needs numeric y)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

# Start with RBF kernel — good default for non-linear medical data
svm_model = SVC(
    kernel="rbf",       # options: 'linear', 'rbf', 'poly'
    C=1.0,              # regularization — higher = less regularization
    gamma="scale",      # auto-scales based on feature variance
    class_weight="balanced", 
    decision_function_shape="ovr",   # one-vs-rest for multiclass
    random_state=42
)

svm_model.fit(X_train, y_train_enc)

#step 5 : Evaluate
from sklearn.metrics import classification_report, accuracy_score

y_pred_enc = svm_model.predict(X_test)

# Decode back to disease names
y_pred = le.inverse_transform(y_pred_enc)
y_test_labels = le.inverse_transform(y_test_enc)

print("Accuracy:", accuracy_score(y_test_enc, y_pred_enc))
print()
print(classification_report(y_test_labels, y_pred, zero_division=0))

#step 6: hyperparameter tuning 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

param_grid = {
    "kernel": ["linear", "rbf", "poly"],
    "C":      [0.1, 1, 10, 100],
    "gamma":  ["scale", "auto"]   # only used by rbf/poly
}

grid_search = GridSearchCV(
    SVC(decision_function_shape="ovr", random_state=42),
    param_grid,
    cv=3,                   # 5-fold cross-validation
    scoring="accuracy",
    n_jobs=-1,              # use all CPU cores
    verbose=1
)

grid_search.fit(X_train, y_train_enc)

print("Best params:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)

# Refit with best model
best_svm = grid_search.best_estimator_

# Evaluate best model on test set
y_pred_best = best_svm.predict(X_test)
print("Best SVM Test Accuracy:", accuracy_score(y_test_enc, y_pred_best))
print(classification_report(
    le.inverse_transform(y_test_enc),
    le.inverse_transform(y_pred_best),
    zero_division=0
))

#Step 7: careplan recommendation logic 
# Build mapping: pathology → most common careplans
careplans_raw = pd.read_csv("/Users/saket/Desktop/BDA/careplans.csv")
symptoms_raw  = pd.read_csv("/Users/saket/Desktop/BDA/symptoms.csv")

# case mismatch resolved
symptoms_raw["PATHOLOGY"] = symptoms_raw["PATHOLOGY"].astype(str).str.strip().str.lower()
careplans_raw["DESCRIPTION"] = careplans_raw["DESCRIPTION"].astype(str).str.strip().str.lower()


# Get one pathology per patient (most frequent)
patient_pathology = (
    symptoms_raw.groupby("PATIENT")["PATHOLOGY"]
    .agg(lambda x: x.value_counts().idxmax())
    .reset_index()
    .rename(columns={"PATHOLOGY": "PRIMARY_PATHOLOGY"})
)

# Join careplans with patient's pathology
merged = careplans_raw.merge(patient_pathology, on="PATIENT", how="left")

# Build frequency map: pathology → ranked list of careplans
pathology_careplan_map = (
    merged.groupby(["PRIMARY_PATHOLOGY", "DESCRIPTION"])
    .size()
    .reset_index(name="count")
    .sort_values(["PRIMARY_PATHOLOGY", "count"], ascending=[True, False])
)

def get_recommended_careplans(pathology, top_n=3):
    recs = pathology_careplan_map[
        pathology_careplan_map["PRIMARY_PATHOLOGY"] == pathology
    ]["DESCRIPTION"].head(top_n).tolist()
    
    if not recs:
        # fallback: most common careplans overall
        recs = careplans_raw["DESCRIPTION"].value_counts().head(top_n).index.tolist()
    return recs

#step 8: full prediction
def predict_and_recommend(patient_features_row, top_n=3):
    """
    patient_features_row: a single row from X (as DataFrame)
    Returns predicted pathology + recommended careplans
    """
    pred_enc = best_svm.predict(patient_features_row)
    predicted_pathology = le.inverse_transform(pred_enc)[0]
    
    care_plans = get_recommended_careplans(predicted_pathology, top_n)
    
    print(f"Predicted condition: {predicted_pathology}")
    print(f"Recommended care plans:")
    for cp in care_plans:
        print(f"  - {cp}")
    
    return predicted_pathology, care_plans

# Example: predict and recommend on test set (for a test patient)
print("\n=== Sample Predictions ===")
for i in range(min(5, len(X_test))):
    sample = X_test.iloc[[i]]
    predict_and_recommend(sample)
    print()
    
    print(pathology_careplan_map.head(15))