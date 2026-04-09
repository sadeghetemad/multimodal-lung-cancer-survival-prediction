import pandas as pd
import time
import numpy as np
from radiomics import featureextractor
import boto3
import os


# ==============================
# AWS Clients (safe for container)
# ==============================
region = os.environ.get("AWS_DEFAULT_REGION", "eu-west-2")

sagemaker_client = boto3.client("sagemaker", region_name=region)
featurestore_runtime = boto3.client("sagemaker-featurestore-runtime", region_name=region)


# ==============================
# Utils
# ==============================
def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == 'object':
            data_frame[label] = data_frame[label].astype("str").astype("string")


# ==============================
# Radiomics
# ==============================
def compute_features(imageName, maskName):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    featureVector = extractor.execute(imageName, maskName)

    new_dict = {}

    for featureName in featureVector.keys():
        val = featureVector[featureName]

        if isinstance(val, np.ndarray):
            new_dict[featureName] = float(val)
        else:
            new_dict[featureName] = val

    df = pd.DataFrame.from_dict(new_dict, orient='index').T
    df = df.convert_dtypes(convert_integer=False)

    df['imageName'] = imageName
    df['maskName'] = maskName

    return df


# ==============================
# Feature Store (BOTO3 version)
# ==============================
def check_feature_group(feature_group_name):
    try:
        response = sagemaker_client.describe_feature_group(
            FeatureGroupName=feature_group_name
        )
        status = response["FeatureGroupStatus"]

        if status == "Created":
            return True
        else:
            wait_for_feature_group_creation_complete(feature_group_name)
            return True

    except sagemaker_client.exceptions.ResourceNotFound:
        return False


def create_feature_group(feature_group_name, dataframe, s3uri,
                         record_id='Subject', event_time='EventTime',
                         enable_online_store=True):

    feature_definitions = []

    for col, dtype in dataframe.dtypes.items():
        if "float" in str(dtype):
            feature_type = "Fractional"
        elif "int" in str(dtype):
            feature_type = "Integral"
        else:
            feature_type = "String"

        feature_definitions.append({
            "FeatureName": col,
            "FeatureType": feature_type
        })

    role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")

    if role_arn is None:
        raise Exception("SAGEMAKER_ROLE_ARN environment variable not set")

    sagemaker_client.create_feature_group(
        FeatureGroupName=feature_group_name,
        RecordIdentifierFeatureName=record_id,
        EventTimeFeatureName=event_time,
        FeatureDefinitions=feature_definitions,
        OnlineStoreConfig={"EnableOnlineStore": enable_online_store},
        OfflineStoreConfig={
            "S3StorageConfig": {"S3Uri": s3uri}
        },
        RoleArn=role_arn
    )

    wait_for_feature_group_creation_complete(feature_group_name)

    return True


def wait_for_feature_group_creation_complete(feature_group_name):
    while True:
        status = sagemaker_client.describe_feature_group(
            FeatureGroupName=feature_group_name
        )["FeatureGroupStatus"]

        if status == "Created":
            print(f"FeatureGroup {feature_group_name} created.")
            break
        elif status == "CreateFailed":
            raise RuntimeError(f"Failed to create FeatureGroup {feature_group_name}")

        print("Waiting for Feature Group...")
        time.sleep(5)


def ingest_to_feature_store(feature_group_name, df):
    records = []

    for _, row in df.iterrows():
        record = []
        for col in df.columns:
            record.append({
                "FeatureName": col,
                "ValueAsString": str(row[col])
            })
        records.append(record)

    for record in records:
        featurestore_runtime.put_record(
            FeatureGroupName=feature_group_name,
            Record=record
        )