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
# BUILD CT UID MAP
# ======================
def build_ct_uid_map(ct_files):
    ct_map = {}
    for f in ct_files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
            ct_map[dcm.SOPInstanceUID] = f
        except:
            continue
    return ct_map


# ======================
# LOAD CT BY UID ORDER
# ======================
def load_ct_from_uids(ct_map, ref_uids):
    slices = []

    for uid in ref_uids:
        if uid in ct_map:
            dcm = pydicom.dcmread(ct_map[uid])
            if hasattr(dcm, 'ImagePositionPatient'):
                slices.append(dcm)

    if len(slices) < 2:
        raise ValueError("Not enough matched CT slices")

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices], axis=-1)

    px = slices[0].PixelSpacing
    z = abs(
        float(slices[1].ImagePositionPatient[2]) -
        float(slices[0].ImagePositionPatient[2])
    )

    affine = np.eye(4)
    affine[0, 0] = float(px[0])
    affine[1, 1] = float(px[1])
    affine[2, 2] = z

    return volume, affine


# ======================
# LOAD SEG + UID LIST
# ======================
def load_seg_with_refs(seg_file):
    dcm = pydicom.dcmread(seg_file)

    seg = dcm.pixel_array

    if seg.ndim == 3:
        seg = np.transpose(seg, (1, 2, 0))

    ref_uids = []

    try:
        ref_seq = dcm.ReferencedSeriesSequence[0].ReferencedInstanceSequence
        for ref in ref_seq:
            ref_uids.append(ref.ReferencedSOPInstanceUID)
    except:
        raise ValueError("SEG has no references")

    return seg.astype(np.uint8), ref_uids


# ======================
# FIND VALID STUDY
# ======================
def find_valid_pair(studies):

    print("🔍 Total studies:", len(studies))

    for study_uid in sorted(studies.keys()):

        data = studies[study_uid]

        if len(data["CT"]) == 0 or len(data["SEG"]) == 0:
            continue

        print("\n🧠 Checking study:", study_uid)

        # ===== LOAD SEG =====
        try:
            seg_vol, ref_uids = load_seg_with_refs(data["SEG"][0])
        except Exception as e:
            print("❌ SEG load failed:", e)
            continue

        print("SEG slices (UID):", len(ref_uids))
        print("SEG shape:", seg_vol.shape)

        # ===== LOAD CT =====
        try:
            ct_map = build_ct_uid_map(data["CT"])
            ct_vol, affine = load_ct_from_uids(ct_map, ref_uids)
        except Exception as e:
            print("❌ CT load failed:", e)
            continue

        print("CT shape:", ct_vol.shape)

        # ===== CHECK RESOLUTION =====
        if ct_vol.shape[0] != seg_vol.shape[0] or ct_vol.shape[1] != seg_vol.shape[1]:
            print("❌ Skip (resolution mismatch)")
            continue

        # ===== CHECK MIN VALID =====
        if ct_vol.shape[2] < 10 or seg_vol.shape[2] < 10:
            print("❌ Skip (too few slices)")
            continue

        # ===== FIX Z =====
        min_slices = min(ct_vol.shape[2], seg_vol.shape[2])

        ct_vol = ct_vol[:, :, :min_slices]
        seg_vol = seg_vol[:, :, :min_slices]

        print("✅ FINAL SELECTED:", ct_vol.shape)

        return ct_vol, seg_vol, affine

    print("\n🚨 DEBUG: No valid pair found after checking all studies")

    for study_uid in studies:
        print("Study:", study_uid,
              "CT:", len(studies[study_uid]["CT"]),
              "SEG:", len(studies[study_uid]["SEG"]))

    raise Exception("No valid CT-SEG pair found")

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

    all_dcms = glob(os.path.join(data_dir, "**", "*.dcm"), recursive=True)
    studies = group_by_study(all_dcms)

    ct_volume, seg_volume, affine = find_valid_pair(studies)

    print("Final shape:", ct_volume.shape)

    # ======================
    # VALIDATE MASK
    # ======================
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

    if not utils.check_feature_group(args.feature_group_name):
        utils.create_feature_group(
            args.feature_group_name,
            df,
            args.offline_store_s3uri
        )

    utils.ingest_to_feature_store(args.feature_group_name, df)

    print("DONE:", args.subject)