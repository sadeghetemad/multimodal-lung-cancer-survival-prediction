import boto3
import pandas as pd
import numpy as np
import joblib
import time
import os
import tarfile
import shutil

import sagemaker
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.image_uris import retrieve
from sagemaker.serializers import CSVSerializer
from sagemaker.session import TrainingInput

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score,  precision_score, recall_score, confusion_matrix

# ==============================
# CONFIG
# ==============================

REGION = "eu-west-2"
BUCKET = "multimodal-lung-cancer-811165582441-eu-west-2-an"
PREFIX = "multi-model-health-ml"
ROLE = "arn:aws:iam::811165582441:role/lung_cancer_diagnostic"

GENOMIC_FG = "genomic-feature-group-05-19-10-59"
CLINICAL_FG = "clinical-feature-group-05-18-48-56"
IMAGING_FG = "ct-seg-image-imaging-feature-group"


# ==============================
# AWS SESSION
# ==============================

def create_session():
    boto_session = boto3.Session(region_name=REGION)

    sm_client = boto_session.client("sagemaker")
    fs_runtime = boto_session.client("sagemaker-featurestore-runtime")

    return Session(
        boto_session=boto_session,
        sagemaker_client=sm_client,
        sagemaker_featurestore_runtime_client=fs_runtime
    )


# ==============================
# LOAD FEATURES (ATHENA JOIN)
# ==============================

def get_multimodal_features(session):

    genomic_fg = FeatureGroup(GENOMIC_FG, session)
    clinical_fg = FeatureGroup(CLINICAL_FG, session)
    imaging_fg = FeatureGroup(IMAGING_FG, session)

    gq = genomic_fg.athena_query()
    cq = clinical_fg.athena_query()
    iq = imaging_fg.athena_query()

    genomic_table = gq.table_name
    clinical_table = cq.table_name
    imaging_table = iq.table_name

    output = f"s3://{BUCKET}/{PREFIX}/queries"

    query = f"""
    SELECT "{genomic_table}".*, "{clinical_table}".*, "{imaging_table}".*
    FROM "{genomic_table}"
    LEFT JOIN "{clinical_table}"
    ON "{clinical_table}".case_id = "{genomic_table}".case_id
    LEFT JOIN "{imaging_table}"
    ON "{clinical_table}".case_id = "{imaging_table}".subject
    """

    gq.run(query_string=query, output_location=output)
    gq.wait()

    df = gq.as_dataframe()

    # drop useless cols
    drop_features = ['case_id', 'case_id.1', 
                  'eventtime', 'write_time', 'api_invocation_time', 'is_deleted',
                  'eventtime.1', 'write_time.1', 'api_invocation_time.1', 'is_deleted.1', 
                  'eventtime.2', 'write_time.2', 'api_invocation_time.2', 'is_deleted.2']
    drop_features_img = ['imagename', 'maskname', 'subject']
    drop_features_img += [i for i in df.columns.tolist() if 'diagnostics' in i]
    df = df.drop(drop_features + drop_features_img, axis=1, errors="ignore")

    return df


# ==============================
# PREPROCESS
# ==============================

def preprocess(df):
    X = df.drop("survivalstatus", axis=1)
    y = df["survivalstatus"]

    X = X.fillna(0)

    return X, y


# ==============================
# SCALER + PCA
# ==============================

def apply_scale_pca(X_train, X_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95, random_state=0)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return scaler, pca, X_train_pca, X_test_pca


# ==============================
# SAVE ARTIFACTS
# ==============================

def save_artifact(obj, name):
    # create local folder
    os.makedirs("artifacts", exist_ok=True)

    # correct path
    path = os.path.join("artifacts", name)

    # save locally
    joblib.dump(obj, path)

    # upload to S3
    s3 = boto3.client("s3", region_name=REGION)
    s3.upload_file(path, BUCKET, f"{PREFIX}/artifacts/{name}")

    print(f"Saved {name} locally and uploaded to S3")


# ==============================
# TRAIN MODEL
# ==============================

def train_model(train_df, val_df):

    print("Starting training...")

    container = retrieve("xgboost", region=REGION, version="1.2-1")

    estimator = sagemaker.estimator.Estimator(
        container,
        ROLE,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        output_path=f"s3://{BUCKET}/{PREFIX}/output",
        base_job_name="xgb-training",
        sagemaker_session=sagemaker.Session()
    )

    estimator.set_hyperparameters(
        eta=0.1,
        objective="binary:logistic",
        num_round=50
    )

    os.makedirs("data", exist_ok=True)

    train_path = os.path.join("data", "train.csv")
    val_path = os.path.join("data", "val.csv")

    train_df.to_csv(train_path, header=False, index=False)
    val_df.to_csv(val_path, header=False, index=False)

    s3 = boto3.client("s3", region_name=REGION)

    s3.upload_file(train_path, BUCKET, f"{PREFIX}/train/train.csv")
    s3.upload_file(val_path, BUCKET, f"{PREFIX}/val/val.csv")

    train_input = TrainingInput(
        f"s3://{BUCKET}/{PREFIX}/train/train.csv",
        content_type="text/csv"
    )

    val_input = TrainingInput(
        f"s3://{BUCKET}/{PREFIX}/val/val.csv",
        content_type="text/csv"
    )

    estimator.fit({
        "train": train_input,
        "validation": val_input
    })

    print("Training completed.")

    return estimator


# ==============================
# DEPLOY
# ==============================

def deploy_model(estimator):

    endpoint_name = f"{PREFIX}-endpoint"

    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        serializer=CSVSerializer(),
        endpoint_name=endpoint_name
    )

    return predictor, endpoint_name


# ==============================
# TEST MODEL
# ==============================

def evaluate(predictor, X_test, y_test):

    preds = predictor.predict(X_test).decode("utf-8")
    preds = [1 if float(p) > 0.5 else 0 for p in preds.split(",")]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, average='weighted'))
    print("Recall:", recall_score(y_test, preds, average='weighted'))
    print("Confusion Matrix:", confusion_matrix(y_test, preds))


# ==============================
# SAVE TRAINED MODEL
# ==============================
def save_trained_model(estimator):

    print("Downloading trained model...")    

    s3 = boto3.client("s3", region_name=REGION)

    model_s3_path = estimator.model_data

    bucket = model_s3_path.split("/")[2]
    key = "/".join(model_s3_path.split("/")[3:])

    os.makedirs("artifacts", exist_ok=True)

    tar_path = "artifacts/model.tar.gz"
    s3.download_file(bucket, key, tar_path)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path="artifacts")

    src = "artifacts/xgboost-model"
    dst = "artifacts/xgboost_model.bin"

    shutil.copy(src, dst)

    print("Model saved as raw binary ✅")

# ==============================
# MAIN PIPELINE
# ==============================

def run():

    print("Starting full pipeline...")

    session = create_session()

    # 1. load data
    df = get_multimodal_features(session)

    # 2. preprocess
    X, y = preprocess(df)

    # save feature order
    feature_order = list(X.columns)
    save_artifact(feature_order, "feature_order.joblib")

    # 3. split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=0
    )

    # 4. PCA
    scaler, pca, X_train_pca, X_test_pca = apply_scale_pca(X_train, X_test)
    X_val_pca = pca.transform(scaler.transform(X_val))

    # 5. save artifacts
    save_artifact(scaler, "scaler.joblib")
    save_artifact(pca, "pca.joblib")

    # 6. prepare training data
    train_df = pd.concat([y_train.reset_index(drop=True), pd.DataFrame(X_train_pca)], axis=1)
    val_df = pd.concat([y_val.reset_index(drop=True), pd.DataFrame(X_val_pca)], axis=1)

    # 7. train and save model
    estimator = train_model(train_df, val_df)
    save_trained_model(estimator)

    # 8. deploy
    predictor, endpoint = deploy_model(estimator)

    # 9. test
    evaluate(predictor, X_test_pca, y_test)

    print("DONE")
    print("Endpoint:", endpoint)


if __name__ == "__main__":
    run()