#!/usr/bin/env python

import argparse
from glob import glob
import os
import numpy as np
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
import time
import radiomics_utils as utils


# ======================
# GROUP BY STUDY
# ======================
def group_by_study(dcm_files):
    studies = {}

    for f in dcm_files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
            study_uid = dcm.StudyInstanceUID

            if study_uid not in studies:
                studies[study_uid] = {"CT": [], "SEG": []}

            if dcm.Modality == "CT":
                studies[study_uid]["CT"].append(f)

            elif dcm.Modality == "SEG":
                studies[study_uid]["SEG"].append(f)

        except:
            continue

    return studies


# ======================
# LOAD CT (STABLE)
# ======================
def load_ct(ct_files):
    slices = []

    for f in ct_files:
        dcm = pydicom.dcmread(f)

        if hasattr(dcm, 'ImagePositionPatient'):
            slices.append(dcm)

    if len(slices) < 2:
        raise ValueError("Not enough CT slices")

    # sort by Z
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices], axis=-1)

    # spacing
    px = slices[0].PixelSpacing
    z = abs(
        float(slices[1].ImagePositionPatient[2]) -
        float(slices[0].ImagePositionPatient[2])
    )

    spacing = [float(px[0]), float(px[1]), z]

    # affine
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]

    return volume, affine


# ======================
# LOAD SEG
# ======================
def load_seg(seg_file):
    dcm = pydicom.dcmread(seg_file)

    seg = dcm.pixel_array

    if seg.ndim == 3:
        seg = np.transpose(seg, (1, 2, 0))

    return seg.astype(np.uint8)


# ======================
# FIND VALID STUDY
# ======================
def find_valid_pair(studies):

    for study_uid, data in studies.items():

        if len(data["CT"]) == 0 or len(data["SEG"]) == 0:
            continue

        print("\nChecking study:", study_uid)

        ct_vol, affine = load_ct(data["CT"])
        seg_vol = load_seg(data["SEG"][0])

        print("CT:", ct_vol.shape, "| SEG:", seg_vol.shape)

        if ct_vol.shape == seg_vol.shape:
            print("✅ MATCH FOUND")
            return ct_vol, seg_vol, affine

    raise Exception("No matching CT-SEG pair found")


# ======================
# MAIN
# ======================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str)
    parser.add_argument("--feature_group_name", type=str)
    parser.add_argument("--offline_store_s3uri", type=str)
    args = parser.parse_args()

    data_dir = "/opt/ml/processing/input/"
    output_dir = "/opt/ml/processing/output/"

    os.makedirs(os.path.join(output_dir, "PNG"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "CT-Nifti"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "CT-SEG"), exist_ok=True)

    # ======================
    # LOAD FILES
    # ======================
    all_dcms = glob(os.path.join(data_dir, "**", "*.dcm"), recursive=True)

    studies = group_by_study(all_dcms)

    # ======================
    # FIND MATCH
    # ======================
    ct_volume, seg_volume, affine = find_valid_pair(studies)

    print("Final shape:", ct_volume.shape)

    # ======================
    # VALIDATE MASK
    # ======================
    unique_vals = np.unique(seg_volume)
    print("SEG unique values:", unique_vals)

    if np.max(seg_volume) == 0:
        raise ValueError("SEGMENTATION IS EMPTY")

    # ======================
    # SAVE
    # ======================
    ct_nii = nib.Nifti1Image(ct_volume, affine)
    seg_nii = nib.Nifti1Image(seg_volume, affine)

    image_path = os.path.join(output_dir, "CT-Nifti", f"{args.subject}.nii.gz")
    mask_path = os.path.join(output_dir, "CT-SEG", f"{args.subject}.nii.gz")

    nib.save(ct_nii, image_path)
    nib.save(seg_nii, mask_path)

    # ======================
    # VIS
    # ======================
    f = plt.figure(figsize=(12, 5))
    plotting.plot_roi(seg_nii, bg_img=ct_nii, figure=f, alpha=0.4)
    plt.savefig(os.path.join(output_dir, "PNG", f"{args.subject}.png"))

    # ======================
    # RADIOMICS
    # ======================
    print("Computing radiomics...")

    df = utils.compute_features(image_path, mask_path)

    df["Subject"] = args.subject
    df["EventTime"] = float(time.time())

    utils.cast_object_to_string(df)

    print("FEATURES:", df.shape)

    if not utils.check_feature_group(args.feature_group_name):
        utils.create_feature_group(
            args.feature_group_name,
            df,
            args.offline_store_s3uri
        )

    utils.ingest_to_feature_store(args.feature_group_name, df)

    print("DONE:", args.subject)