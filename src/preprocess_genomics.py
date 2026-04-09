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
DATA_KEY = "GSE103584_R01_NSCLC_RNAseq.txt"
PREFIX = "sagemaker-featurestore-demo"


SELECTED_COLUMNS = [
    'Case_ID','LRIG1','HPGD','GDF15','CDH2','POSTN','VCAN','PDGFRA',
    'VCAM1','CD44','CD48','CD4','LYL1','SPI1','CD37','VIM','LMO2',
    'EGR2','BGN','COL4A1','COL5A1','COL5A2'
]

DROP_CASES = [
    'R01-003','R01-004','R01-006','R01-007','R01-015',
    'R01-016','R01-018','R01-022','R01-023','R01-098','R01-105'
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

    return feature_store_session, region


# ==============================
# LOAD DATA
# ==============================

def load_genomic_data():
    data_location = f"s3://{BUCKET_NAME}/{DATA_KEY}"
    df = pd.read_csv(data_location, delimiter="\t")
    return df


# ==============================
# PREPROCESS
# ==============================

def preprocess_genomic_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unwanted cases
    df = df.drop(DROP_CASES, axis=1)

    # Rename index column
    df.rename(columns={'Unnamed: 0': 'index_temp'}, inplace=True)

    # Transpose
    df.set_index('index_temp', inplace=True)
    df = df.transpose().reset_index()
    df.rename(columns={'index': 'Case_ID'}, inplace=True)

    # Select features
    df = df[SELECTED_COLUMNS]

    # Fill NaN
    df = df.fillna(0)

    return df


# ==============================
# FEATURE STORE HELPERS
# ==============================

def cast_object_to_string(df: pd.DataFrame):
    for col in df.columns:
        if df.dtypes[col] == "object":
            df[col] = df[col].astype("string")


def wait_for_feature_group(feature_group: FeatureGroup):
    status = feature_group.describe().get("FeatureGroupStatus")

    while status == "Creating":
        print("Waiting for Feature Group creation...")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")

    if status != "Created":
        raise RuntimeError(f"FeatureGroup creation failed: {status}")

    print(f"FeatureGroup {feature_group.name} created successfully.")


# ==============================
# CREATE FEATURE GROUP
# ==============================

def create_feature_group(df: pd.DataFrame, feature_store_session: Session):
    feature_group_name = "genomic-feature-group-" + strftime('%d-%H-%M-%S', gmtime())

    fg = FeatureGroup(name=feature_group_name, sagemaker_session=feature_store_session)

    # Cast types
    cast_object_to_string(df)

    # Add event time
    current_time = int(time.time())
    df["EventTime"] = pd.Series([current_time] * len(df), dtype="float64")

    # Load schema
    fg.load_feature_definitions(data_frame=df)

    # Create feature group
    default_bucket = feature_store_session.default_bucket()

    fg.create(
        s3_uri=f"s3://{default_bucket}/{PREFIX}",
        record_identifier_name="Case_ID",
        event_time_feature_name="EventTime",
        role_arn=ROLE_ARN,
        enable_online_store=True
    )

    wait_for_feature_group(fg)

    # Ingest data
    fg.ingest(data_frame=df, max_workers=3, wait=True)

    return fg


# ==============================
# MAIN PIPELINE
# ==============================

def run_pipeline():
    print("Starting genomic preprocessing pipeline...")

    feature_store_session, _ = create_sessions()

    df_raw = load_genomic_data()
    df_processed = preprocess_genomic_data(df_raw)

    fg = create_feature_group(df_processed, feature_store_session)

    print("Pipeline completed successfully.")
    print(f"Feature Group Name: {fg.name}")


# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    run_pipeline()