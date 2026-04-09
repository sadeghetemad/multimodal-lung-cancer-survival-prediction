import boto3
import pandas as pd
import time
from time import gmtime, strftime
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup


# ==============================
# CONFIG
# ==============================

ROLE_ARN = "arn:aws:iam::811165582441:role/lung_cancer_diagnostic"
BUCKET_NAME = "nsclc-clinical-genomic-data-811165582441-eu-west-2-an"
DATA_KEY = "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
PREFIX = "sagemaker-featurestore-demo"


ENCODE_COLS = [
    "Patient affiliation", "Gender", "Ethnicity", "Smoking status", "%GG",
    "Tumor Location (choice=RUL)", "Tumor Location (choice=RML)",
    "Tumor Location (choice=RLL)", "Tumor Location (choice=LUL)",
    "Tumor Location (choice=LLL)", "Tumor Location (choice=L Lingula)",
    "Tumor Location (choice=Unknown)", "Histology ",
    "Pathological T stage", "Pathological N stage", "Pathological M stage",
    "Histopathological Grade", "Lymphovascular invasion",
    "Pleural invasion (elastic, visceral, or parietal)",
    "EGFR mutation status", "KRAS mutation status",
    "ALK translocation status", "Adjuvant Treatment",
    "Chemotherapy", "Radiation", "Recurrence", "Recurrence Location"
]

NUMERIC_COLS = [
    "Case ID", "Age at Histological Diagnosis", "Weight (lbs)",
    "Pack Years", "Time to Death (days)",
    "Days between CT and surgery", "Survival Status"
]

DROP_COLS = [
    "Quit Smoking Year", "Date of Recurrence",
    "Date of Last Known Alive", "Date of Death",
    "CT Date", "PET Date"
]


# ==============================
# AWS SESSION
# ==============================

def create_sessions():
    region = boto3.Session().region_name

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    featurestore_runtime = boto_session.client("sagemaker-featurestore-runtime")

    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime
    )

    return feature_store_session


# ==============================
# LOAD DATA
# ==============================

def load_clinical_data():
    data_location = f"s3://{BUCKET_NAME}/{DATA_KEY}"
    df = pd.read_csv(data_location)
    return df


# ==============================
# PREPROCESS
# ==============================

def preprocess_clinical_data(df: pd.DataFrame) -> pd.DataFrame:

    # Keep only R01 cases
    df = df[~df["Case ID"].str.contains("AMC")]

    # Drop useless columns
    df = df.drop(DROP_COLS, axis=1)

    # One-hot encoding
    df_encoded = pd.get_dummies(df[ENCODE_COLS])
    df_numeric = df[NUMERIC_COLS]

    df = pd.concat([df_encoded, df_numeric], axis=1)

    # Clean column names
    df = clean_column_names(df)

    # Fix label
    df["SurvivalStatus"].replace({"Dead": "1", "Alive": "0"}, inplace=True)

    # Remove bad samples
    df = df[df['Weightlbs'] != "Not Collected"]
    df = df[df['PackYears'] != "Not Collected"]

    # Fill NaN
    df = df.fillna(0)

    # Convert bool to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    invalid_chars = ['-',' ','%','/','<','>','(',')','=',',',':']

    new_columns = {}
    for col in df.columns:
        new_col = col

        if col == "Case ID":
            new_col = col.replace(" ", "_")
        else:
            for ch in invalid_chars:
                new_col = new_col.replace(ch, "")

            if len(new_col) >= 64:
                new_col = new_col[:60]

        new_columns[col] = new_col

    return df.rename(columns=new_columns)


# ==============================
# FEATURE STORE
# ==============================

def cast_object_to_string(df: pd.DataFrame):
    for col in df.columns:
        if df.dtypes[col] == "object":
            df[col] = df[col].astype("string")


def wait_for_feature_group(feature_group: FeatureGroup):
    status = feature_group.describe().get("FeatureGroupStatus")

    while status == "Creating":
        print("Waiting for Feature Group...")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")

    if status != "Created":
        raise RuntimeError(f"FeatureGroup failed: {status}")

    print(f"{feature_group.name} created successfully")


def create_feature_group(df: pd.DataFrame, session: Session):
    feature_group_name = "clinical-feature-group-" + strftime('%d-%H-%M-%S', gmtime())

    fg = FeatureGroup(name=feature_group_name, sagemaker_session=session)

    cast_object_to_string(df)

    # Add event time
    current_time = int(time.time())
    df["EventTime"] = current_time

    df["EventTime"] = df["EventTime"].fillna(0)

    fg.load_feature_definitions(data_frame=df)

    bucket = session.default_bucket()

    fg.create(
        s3_uri=f"s3://{bucket}/{PREFIX}",
        record_identifier_name="Case_ID",
        event_time_feature_name="EventTime",
        role_arn=ROLE_ARN,
        enable_online_store=True
    )

    wait_for_feature_group(fg)

    fg.ingest(data_frame=df, max_workers=3, wait=True)

    return fg


# ==============================
# PIPELINE
# ==============================

def run_pipeline():
    print("Starting clinical preprocessing pipeline...")

    session = create_sessions()

    df_raw = load_clinical_data()
    df_processed = preprocess_clinical_data(df_raw)

    fg = create_feature_group(df_processed, session)

    print("Pipeline completed.")
    print(f"Feature Group: {fg.name}")


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    run_pipeline()