import pandas as pd
import numpy as np
import ast

# load the files 
careplans = pd.read_csv("../careplans.csv")
symptoms = pd.read_csv("../symptoms.csv")
patients = pd.read_csv("../patients.csv")

print("careplans columns:", careplans.columns.tolist())
print("symptoms columns:", symptoms.columns.tolist())
print("patients columns:", patients.columns.tolist())

# clean the IDs and ensure they match across datasets
patients["Id"] = patients["Id"].astype(str).str.strip()
symptoms["PATIENT"] = symptoms["PATIENT"].astype(str).str.strip()
careplans["PATIENT"] = careplans["PATIENT"].astype(str).str.strip()

valid_ids = set(patients["Id"])

symptoms = symptoms[symptoms["PATIENT"].isin(valid_ids)].copy()
careplans = careplans[careplans["PATIENT"].isin(valid_ids)].copy()

print("\nExpected patients:", len(valid_ids))
print("Unique patients in symptoms:", symptoms["PATIENT"].nunique())
print("Unique patients in careplans:", careplans["PATIENT"].nunique())

# sanity checks
print("SANITY CHECKS")
print("Duplicate rows in symptoms:", symptoms.duplicated().sum())
print("Duplicate rows in careplans:", careplans.duplicated().sum())
print("Duplicate IDs in patients:", patients["Id"].duplicated().sum())

print("\nMissing values in symptoms:")
print(symptoms.isna().sum())

print("\nMissing values in careplans:")
print(careplans.isna().sum())

symptom_ids = set(symptoms["PATIENT"])
careplan_ids = set(careplans["PATIENT"])

print("\nPatients missing careplans:", len(symptom_ids - careplan_ids))
print("Patients missing symptoms:", len(careplan_ids - symptom_ids))

# drop exact duplicate rows in symptoms
symptoms = symptoms.drop_duplicates().copy()

# clean text fields/strip them 
symptoms["PATHOLOGY"] = symptoms["PATHOLOGY"].astype(str).str.strip().str.lower()
symptoms["SYMPTOMS"] = symptoms["SYMPTOMS"].astype(str).str.strip()
careplans["DESCRIPTION"] = careplans["DESCRIPTION"].astype(str).str.strip().str.lower()

# parse the SYMPTOMS field into a list of symptoms
def parse_symptom_list(x):
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            return [str(item).strip().lower() for item in parsed if str(item).strip() != ""]
    except:
        pass

    if ";" in x:
        parts = x.split(";")
    elif "," in x:
        parts = x.split(",")
    else:
        parts = [x]

    return [p.strip().lower() for p in parts if p.strip() != ""]

symptoms["SYMPTOM_LIST"] = symptoms["SYMPTOMS"].apply(parse_symptom_list)

# aggregate to one row per patient, taking first value for demographics and max for NUM_SYMPTOMS
patient_base = symptoms.groupby("PATIENT").agg({
    "GENDER": "first",
    "RACE": "first",
    "ETHNICITY": "first",
    "AGE_BEGIN": "first",
    "AGE_END": "first",
    "PATHOLOGY": "first",
    "NUM_SYMPTOMS": "max"
})

symptom_lists = symptoms.groupby("PATIENT")["SYMPTOM_LIST"].sum()
symptom_lists = symptom_lists.apply(lambda x: sorted(list(set(x))))
patient_base = patient_base.join(symptom_lists)

# create binary features for each unique symptom
all_symptoms = sorted(
    set(symptom for sublist in patient_base["SYMPTOM_LIST"] for symptom in sublist)
)

for symptom in all_symptoms:
    patient_base[f"SYMPTOM__{symptom}"] = patient_base["SYMPTOM_LIST"].apply(
        lambda x: int(symptom in x)
    )

patient_base["NUM_SYMPTOMS_COMPUTED"] = patient_base["SYMPTOM_LIST"].apply(len)

print("\nTotal unique symptom features:", len(all_symptoms))

# aggregate careplans to patient level
careplans = careplans[careplans["DESCRIPTION"] != ""].copy()

careplan_lists = (
    careplans.groupby("PATIENT")["DESCRIPTION"]
    .apply(lambda x: sorted(list(set(x))))
    .rename("CAREPLAN_LIST")
)

patient_base = patient_base.join(careplan_lists)
patient_base["CAREPLAN_LIST"] = patient_base["CAREPLAN_LIST"].apply(
    lambda x: x if isinstance(x, list) else []
)
patient_base["NUM_CAREPLANS"] = patient_base["CAREPLAN_LIST"].apply(len)

all_careplans = sorted(
    set(cp for sublist in patient_base["CAREPLAN_LIST"] for cp in sublist)
)

for cp in all_careplans:
    patient_base[f"CAREPLAN__{cp}"] = patient_base["CAREPLAN_LIST"].apply(
        lambda x: int(cp in x)
    )

print("Total unique careplan labels:", len(all_careplans))

# handle missing values
meta = patient_base.copy()

numeric_fill_cols = ["AGE_BEGIN", "AGE_END", "NUM_SYMPTOMS", "NUM_SYMPTOMS_COMPUTED", "NUM_CAREPLANS"]
for col in numeric_fill_cols:
    if col in meta.columns:
        meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0)

meta["PATHOLOGY"] = meta["PATHOLOGY"].fillna("unknown")
meta["SYMPTOM_LIST"] = meta["SYMPTOM_LIST"].apply(lambda x: x if isinstance(x, list) else [])
meta["CAREPLAN_LIST"] = meta["CAREPLAN_LIST"].apply(lambda x: x if isinstance(x, list) else [])

# encode categorical variables
meta_encoded = pd.get_dummies(
    meta,
    columns=["GENDER", "RACE", "ETHNICITY"],
    dummy_na=True
)

#  normalize numerical features
scale_cols = [c for c in ["AGE_BEGIN", "AGE_END", "NUM_SYMPTOMS", "NUM_SYMPTOMS_COMPUTED", "NUM_CAREPLANS"] if c in meta_encoded.columns]

for col in scale_cols:
    col_min = meta_encoded[col].min()
    col_max = meta_encoded[col].max()
    if col_max > col_min:
        meta_encoded[col] = (meta_encoded[col] - col_min) / (col_max - col_min)
    else:
        meta_encoded[col] = 0

# final checks
print("\n=== FINAL CHECKS ===")
print("Final shape:", meta_encoded.shape)
print("Unique patients in final dataset:", meta_encoded.index.nunique())
print("Missing values in final dataset:", meta_encoded.isna().sum().sum())
print("Duplicate patient rows:", meta_encoded.index.duplicated().sum())

# save outputs
meta.to_csv("../meta_dataset_readable.csv", index=True)
meta_encoded.to_csv("../meta_dataset_ml_ready.csv", index=True)

print("\nSaved files:")
print("../meta_dataset_readable.csv")
print("../meta_dataset_ml_ready.csv")

print("\nPreview:")
print(meta_encoded.head())



import matplotlib.pyplot as plt

symptom_cols = [c for c in meta_encoded.columns if c.startswith("SYMPTOM__")]

symptom_counts = meta_encoded[symptom_cols].sum().sort_values(ascending=False)

symptom_counts.head(15).plot(kind="bar")
plt.title("Top 15 Most Common Symptoms")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()