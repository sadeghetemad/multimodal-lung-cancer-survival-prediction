# 🧠 Multimodal Lung Cancer Survival Prediction Pipeline

An end-to-end **multimodal machine learning system** for lung cancer survival prediction using **clinical, genomic, and CT imaging data**, fully built and orchestrated on **AWS SageMaker**.

---

## 🚀 Overview

This project implements a **cloud-native multimodal pipeline** where all stages — from preprocessing to training and inference — are executed on **AWS SageMaker**.

Three data modalities are used:

- Clinical (CSV)
- Genomic (text / structured)
- CT Imaging (DICOM)

Each modality is transformed into features and stored in **SageMaker Feature Store**, enabling scalable training and querying via **Amazon Athena**.

---

## ☁️ Fully AWS-Native Architecture

Everything in this project runs on AWS:

- **SageMaker Studio Notebooks** → orchestration & preprocessing  
- **SageMaker Processing Jobs** → heavy image pipelines  
- **Docker (custom container)** → radiomics feature extraction  
- **SageMaker Feature Store** → central feature storage  
- **SageMaker Training (XGBoost)** → model training  
- **SageMaker Endpoint** → real-time inference  
- **SageMaker Experiments** → experiment tracking  

This is not a local ML project pretending to scale. It is **designed to scale from day one**.

---

## 🧩 Pipeline Breakdown

### 1. Clinical & Genomic Pipeline

- Processed inside SageMaker notebooks  
- Cleaned and converted into tabular features  
- Directly written to **Feature Store**

---

### 2. Imaging Pipeline (Core Complexity)

CT scans require a full processing pipeline:

1. Load **DICOM CT images + tumor segmentation**
2. Convert slices → **3D volume**
3. Align CT with segmentation (registration)
4. Extract **radiomics features**
5. Convert to tabular format
6. Store in **Image Feature Group**

---

## ⚙️ Scalable Image Processing (Important Part)

This part runs entirely on **SageMaker Processing Jobs** using Docker containers.

- Each patient = one container job
- Parallel processing across subjects
- Heavy computation happens in the cloud, not your laptop

Feature extraction logic lives in:

```
src/
├── dcm2nifti_processing.py
├── preprocess_images.py
├── radiomics_utils.py
```

Dockerized via:

```
src/Dockerfile
```

---

## 🏗️ Feature Store Design

Each modality has its own Feature Group:

- Clinical Feature Group  
- Genomic Feature Group  
- Imaging Feature Group  

All features are:

- Stored centrally  
- Queryable via Athena  
- Joinable for training  

---

## 🤖 Model Training

- Features retrieved from Feature Store  
- Combined into a unified dataset  
- Trained using **SageMaker XGBoost**  

Notebook:

```
4_train_test_model.ipynb
```

---

## ⚡ Inference

- Model deployed to **SageMaker Endpoint**
- Supports real-time predictions

---

## 🧪 Project Structure

```
notebooks/
├── 1_preprocess_genomic_data.ipynb
├── 2_preprocess_clinical_data.ipynb
├── 3_preprocess_imaging_data.ipynb
├── 4_train_test_model.ipynb

src/
├── dcm2nifti_processing.py
├── preprocess_images.py
├── radiomics_utils.py
├── Dockerfile
├── requirements.txt
└── step_function.json

main.py
```

---

## 🔥 Key Highlights

- True multimodal ML (clinical + genomic + imaging)
- Fully AWS-native (no fake local pipelines)
- Scalable CT processing with Docker + SageMaker
- Radiomics feature engineering from 3D volumes
- Feature Store-driven architecture
- Production-ready design (training + endpoint)

---

## 🎯 Goal

To build a **scalable, production-grade AI system** for lung cancer detection by combining multiple medical data sources into a unified learning pipeline.

---

If you’re just stacking notebooks, this repo will feel like overkill.  
If you actually care about building real ML systems, this is the blueprint.
