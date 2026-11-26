# Data Acquisition & Guidelines

This directory contains instructions and guidelines for acquiring the source datasets used in this capstone project. 

Due to user registration requirements, data license agreements, and medical privacy disclaimers, raw imaging files and clinical tables cannot be hosted directly in this repository. Follow the steps below to obtain and structure the data correctly.

---

## 1. Neuro-Imaging Data: RSNA-MICCAI BraTS 2023 Challenge
The imaging dataset comprises multi-parametric 3D NIfTI medical scans sourced from the **RSNA-MICCAI BraTS 2023 (Brain Tumor Segmentation)** challenge repositories via Synapse[cite: 1].

* **Access & Licensing:** Users must register for a free account on [Synapse.org](https://www.synapse.org), navigate to the BraTS 2023 challenge portal, and explicitly agree to the Synapse data use agreement and medical imaging disclaimer before downloading.
* **Dataset Scale:** 1,251 patient 3D NifTI medical image folders[cite: 1].
* **Required Modalities:** Each patient directory contains four co-registered, skull-stripped, and isotropic-resampled modalities:
  * `t1n`: T1-native scans
  * `t1c`: T1-post contrast scans (`t1ce`)
  * `t2w`: T2-weighted scans
  * `t2f`: T2-FLAIR scans
* **Mapping File:** To reconcile patient IDs with cross-institutional metadata, ensure the reference mapping file `BraTS2023_2017_GLI_Mapping.xlsx` is placed in your local data directory.

---

## 2. Clinical & Genomic Metadata: cBioPortal
Clinical outcome data, demographic markers, and genomic mutation profiles were compiled from cBioPortal[cite: 1]. For detailed cohort breakdowns and study selections, please refer to the project presentation slides in the `/docs` directory.

* **Access & Licensing:** Data can be queried and downloaded through the [cBioPortal for Cancer Genomics](https://www.cbioportal.org/) platform under open-access research terms.
* **Selected Cohorts & Studies:** Clinical and molecular data were integrated from the following primary neuro-oncology studies:
  * `gbm_tcga_pan_can_atlas_2018` (TCGA, PanCancer Atlas)[cite: 1]
  * `lgg_tcga_pan_can_atlas_2018` (TCGA, Low-Grade Glioma PanCancer Atlas)[cite: 1]
  * `gbm_cptac_2021` (Clinical Proteomic Tumor Analysis Consortium)[cite: 1]
  * `lggbm_tcga_pub` (TCGA, LGG/GBM Publication Cohort)[cite: 1]
* **Consolidated Output:** The merged and cleaned clinical metadata table is structured as `combined_study_clinical_data.tsv`. This table provides overall survival days, status, progression metrics, patient age, biological sex, KPS, IDH subtype, MGMT promoter status, and genomic instability indices (such as Fraction Genome Altered and tumor mutation counts).
