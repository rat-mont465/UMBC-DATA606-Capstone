# Automated Brain Tumor Segmentation and Multi-Modal Survival & Prognosis Prediction

**Prepared for:** UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang

**Author:** Ryan Alexander Trask

**GitHub Repository:** [https://github.com/rat-mont465/UMBC-DATA606-Capstone/tree/main](https://github.com/rat-mont465/UMBC-DATA606-Capstone/tree/main)

**LinkedIn Profile:** https://www.linkedin.com/in/ryan-trask-72101b2b4/

**PowerPoint Presentation:** [https://github.com/rat-mont465/UMBC-DATA606-Capstone/raw/main/docs/Multimodal%20Brain%20Tumor%20Risk%20and%20Variable%20Assessment_Final.pptx](https://github.com/rat-mont465/UMBC-DATA606-Capstone/raw/main/docs/Multimodal%20Brain%20Tumor%20Risk%20and%20Variable%20Assessment_Final.pptx)

**YouTube Video:**

## 1. Background

### What is it about?

This project focuses on building an end-to-end, fully automated machine learning pipeline for brain tumor medical imaging analysis and clinical outcome prediction. By combining 3D deep learning for automated MRI tumor segmentation with quantitative radiomics feature extraction, deep visual embeddings, and clinical/genomic metadata, the pipeline predicts patient survivability and disease progression risk.

### Why does it matter?

Throughout tumor medical imaging, human analysis runs into two primary issues:

1. **Inter- and Intra-Rater Variability:** A clinician at the beginning of a shift is more rested and capable of identifying critical tumor boundaries than later in the day, and more experienced specialists often achieve better diagnostic consistency for their patients.

2. **Biological Perception Bottlenecks:** Human eyes face inherent limitations in visual image processing. Computational algorithms can detect subtle, sub-visual shape descriptors, voxel intensity distributions, and high-order texture matrices that human observers struggle to perceive.

Automating both the segmentation and biomarker extraction stages bridges the gap left by fatigued or overburdened clinicians, unlocking hidden prognostic patterns that drive individual patient risk.

### Research Questions

1. Can a 3D deep learning model accurately and consistently segment multi-modal brain tumor sub-regions (Enhancing Tumor, Tumor Core, and Whole Tumor) from 3D MRI scans?

2. Can quantitative PyRadiomics and deep visual features extracted from these automated segmentations match or outperform human ground-truth segmentations when predicting patient outcomes?

3. How does combining radiomic biomarkers with clinical and genomic metadata improve predictive performance for Overall Survival (OS) and Progression-Free Survival (PFS) compared to using clinical metadata alone?

## 2. Data

### Data Sources

* **Imaging Dataset:** BraTS 2023 Adult Glioma (GLI) Challenge dataset, consisting of multi-institutional, multi-parametric 3D MRI scans (NIfTI format) and expert-annotated ground-truth segmentation masks.

* **Clinical & Genomic Metadata:** Matched clinical outcome tables and genomic profiles sourced across four glioma studies on **cBioPortal** (such as TCGA-GBM and TCGA-LGG cohorts).

### Data Size & Shape

* **Imaging Cohort:** **1,251 patients**, each with 4 distinct 3D MRI modalities (FLAIR, T1, T1ce, and T2) plus a 3D ground-truth mask volume.

* **Clinically-Matched Subset:** **166 patients** successfully cross-referenced between the BraTS 2023 imaging cohort and cBioPortal clinical/genomic records.

* **Extracted Radiomics Feature Space:** Over **1,000 quantitative features** extracted per patient across sub-regions and modalities, subsequently pruned to the **top 30 non-collinear features** alongside clinical predictors for downstream modeling.

### Time Period

The retrospective multi-institutional imaging and clinical records span multi-year longitudinal clinical trials and genomic sequencing initiatives aggregated up through the 2023 BraTS benchmark release.

### Unit of Observation

Each row in the final tidy modeling tables represents **one unique patient** (`Patient_ID`), linking their aggregated 3D radiomic/deep visual features with their corresponding clinical demographics, genomic alterations, and longitudinal survival outcomes.

### Data Dictionary

| Column Name | Data Type | Definition | Potential Values / Range | 
| ----- | ----- | ----- | ----- | 
| `Patient_ID` | String | Unique anonymized patient identifier linking MRI and clinical records | Alphanumeric ID (e.g., `BraTS-GLI-00001-000`) | 
| `Age_at_Diagnosis` | Float | Patient's age in years at the time of initial glioma diagnosis | Continuous (approx. 18 – 89 years) | 
| `Sex` | Categorical | Biological sex of the patient | `Male`, `Female` | 
| `FGA` | Float | Fraction of Genome Altered (percentage of genome affected by copy number gains/losses) | Continuous (`0.0` to `1.0`) | 
| `Mutation_Count` | Integer | Total number of somatic mutations identified in the tumor sample | Discrete non-negative integers (`0+`) | 
| `TMB` | Float | Tumor Mutational Burden (mutations per megabase) | Continuous non-negative values | 
| `Radiomic_Features` | Float | Top 30 pruned PyRadiomics (Shape, First-Order, GLCM, GLRLM, GLSZM, NGTDM) & 3D deep encoder features | Continuous Z-score standardized values | 
| `OS_Status` | Binary | Overall Survival event indicator (Target 1) | `0` (Living / Censored), `1` (Deceased) | 
| `OS_Days` | Float | Time-to-event in days from diagnosis to death or last follow-up (Target 2) | Continuous positive values (days) | 
| `PFS_Status` | Binary | Progression-Free Survival event indicator (Target 3) | `0` (No Progression / Censored), `1` (Progressed) | 
| `PFS_Days` | Float | Time-to-event in days from diagnosis to disease progression or last follow-up (Target 4) | Continuous positive values (days) | 

### Target Variables & Predictors

* **Targets / Labels:**

  * *Binary Classification Endpoints:* `OS_Status` (survival status) and `PFS_Status` (prognosis/progression status).

  * *Time-to-Event Survival Endpoints:* `OS_Days` (overall survival duration) and `PFS_Days` (progression-free survival duration).

* **Features / Predictors:** Clinical variables (`Age_at_Diagnosis`, `Sex`, missingness indicator flags), genomic metrics (`FGA`, `Mutation_Count`, `TMB`), and the top 30 non-collinear quantitative radiomic and deep visual features.

## 3. Exploratory Data Analysis (EDA)

### Data Exploration & Cleansing

Exploratory data analysis was conducted in Jupyter Notebooks to inspect image integrity, demographic distributions, and missingness patterns:

* **Image Integrity Validation & IBSI Standardization:** All 1,251 3D NIfTI patient folders were programmatically verified for file integrity, spatial dimension alignment, and presence of all four MRI modalities (`FLAIR`, `T1`, `T1ce`, `T2`). To ensure feature reproducibility and comply with **Image Biomarker Standardization Initiative (IBSI)** guidelines, all scans underwent isotropic voxel spacing resampling ($1.0 \times 1.0 \times 1.0\text{ mm}$) and Z-score intensity normalization prior to feature extraction, mitigating scanner-induced radiomic variation.

* **Feature Elimination Due to High `NaN` Counts:** Merging clinical records across four distinct cBioPortal studies revealed severe sparsity across numerous candidate clinical attributes (e.g., treatment regimens, Karnofsky performance scores, and specific molecular subtype annotations). Variables exhibiting prohibitively high `NaN` counts—often missing in the majority of matched patients due to inconsistent reporting across source studies—had to be systematically dropped, as imputing them would have introduced synthetic noise and severe bias.

* **Imputation of Retained Clinical Variables:** Among the viable clinical and genomic variables retained for modeling, moderate missingness remained:

  * `Age_at_Diagnosis`: **37.3% missing**

  * `Sex`: **37.3% missing**

  * `Fraction of Genome Altered (FGA)`: **3.6% missing**

  * These remaining missing values were imputed using **Multiple Imputation by Chained Equations (MICE)** paired with explicit binary imputation indicator flags so downstream models could account for missingness patterns.

### Summary Statistics & Distributions

* **Demographics:** Patient age at diagnosis followed a relatively even, bell-shaped distribution across adult age groups. Patient sex was well-balanced with a slight female skew.

* **Genomic & Target Skewness:** `FGA`, genomic mutation counts, and `TMB` exhibited strong right-skewness. Similarly, time-to-event target variables (`OS_Days` and `PFS_Days`) were right-skewed, reflecting typical clinical survival curves where a subset of patients experiences long-term survival.

### Data Augmentation, Merging & Feature Engineering

* **Multi-Source Merging:** BraTS patient identifiers were mapped and merged with cBioPortal clinical and genomic tables to create a unified, tidy patient-level dataset (`1 row = 1 patient`).

* **Correlation & Multicollinearity Pruning:** Spearman rank correlation heatmaps confirmed strong inter-correlations among genomic variables and clinical survival targets, while individual radiomic features showed weaker yet statistically significant correlations with survival. Because extracting >1,000 PyRadiomics features introduced severe multicollinearity, we applied **Sparse PCA absolute component loadings**, unsupervised variance/normality scoring, and a **greedy collinearity pruning algorithm (**$r > 0.75$ **threshold)**. This retained the **top 30 non-collinear features** in their native scale rather than opaque principal components, preserving clinical explainability.

* **Experimental Datasets Constructed:** To test our core hypothesis, three parallel tidy datasets were prepared:

  1. `Clinical Metadata Only`

  2. `Clinical + Ground-Truth Mask Radiomics`

  3. `Clinical + Predicted Mask Radiomics`

## 4. Model Training

### Development Environment & Python Stack

* **Environments:** Local GPU workstation and cloud compute environments (Jupyter Notebooks).

* **Core Python Libraries:** `PyTorch`, `MONAI` (Medical Open Network for AI), `PyRadiomics`, `scikit-learn`, `scikit-survival` / `lifelines`, `NiBabel`, `NumPy`, `Pandas`, `Plotly`, and `Streamlit`.

### Stage 1: 3D Brain Tumor Segmentation

* **Data Splitting:** From the 1,251 imaging cases, the **166 clinically-matched patients** were isolated as a strict **held-out test set** to prevent any data leakage into survival modeling. The remaining **1,085 patients** were partitioned using **5-fold cross-validation** (80/20 train/validation splits per fold).

* **Preprocessing & Loss Function:** Input MRI volumes were pre-cropped to skull-stripped bounding boxes, resampled to isotropic voxel spacing, Z-score intensity normalized, and augmented via spatial patch cropping. To overcome severe class imbalance (where background voxels vastly outnumber tumor sub-regions), a custom **combined Focal + Dice Loss** function was implemented.

* **Architecture Selection:** After benchmarking MONAI's `Swin UNETR`—which imposed prohibitive computational overhead for marginal gains—we selected **MONAI's 3D `SegResNet`** architecture.

* **Segmentation Performance:**

  * **5-Fold Cross-Validation Mean Dice Score:** **0.8872**

  * **Held-Out Test Set (166 patients):** **Mean Dice = 0.79**, **Median Dice = 0.84** (with minor outliers driven by institutional scanner variations), representing strong performance aligned with BraTS medical imaging literature.

### Stage 2: Survival & Prognosis Predictive Modeling

* **Models Evaluated:**

  * *Time-to-Event Survival (`OS_Days`, `PFS_Days`):* Cox Proportional Hazards (`CoxPH` evaluated across L2 penalties from `0.001` to `0.05`), `DeepSurv` Neural Networks, Random Survival Forests / Regressors, Gradient Boosting Regressors, and Ridge Regression.

  * *Binary Classification (`OS_Status`, `PFS_Status`):* L2-Regularized Logistic Regression, Random Forest Classifiers, Support Vector Machines (SVM), and Gradient Boosting.

* **Training & Validation Strategy:** All tabular models were trained using **5-fold cross-validation** with exhaustive hyperparameter grid search across all three feature sets (`Clinical Only`, `Clinical + Ground Truth`, `Clinical + Predicted`).

* **Evaluation Metrics & Quantitative Results:** Models were compared using **95% confidence interval bootstrapping**, Harrell's Concordance Index (C-index), ROC-AUC, Precision-Recall (PR) curves, probability calibration curves, and **Decision Curve Analysis (DCA)**:

#### 1. Time-to-Event Survival Model Architecture Benchmark (C-Index)

Standard regression and deep survival networks struggled with right-censored time-to-event data—performing below the `0.500` random chance baseline (`Random Forest Regressor`: `0.456` OS / `0.454` PFS; `Ridge Regression`: `0.442` OS / `0.478` PFS; `Gradient Boosting Regressor`: `0.411` OS / `0.433` PFS; `DeepSurv Neural Network`: `0.293` PFS)—whereas **Cox Proportional Hazards (`CoxPH`)** models properly handled censoring and achieved strong discriminative performance across penalty settings:

| Model / Penalty Configuration | Overall Survival (`OS_Days`) C-Index | Progression-Free Survival (`PFS_Days`) C-Index | 
| ----- | ----- | ----- | 
| **Cox PH (`pen=0.001`)** | **0.795** | 0.736 | 
| **Cox PH (`pen=0.005`)** | 0.793 | **0.768** | 
| **Cox PH (`pen=0.01`)** | 0.792 | 0.766 | 
| **Cox PH (`pen=0.02`)** | 0.788 | 0.764 | 
| **Cox PH (`pen=0.05`)** | 0.779 | 0.758 | 

#### 2. Feature Domain Comparison & Multimodal Feature Lift

Comparing the best models across our three feature domains confirmed that integrating automated radiomic biomarkers delivers substantial predictive lift for continuous survival modeling, while binary status classification remains competitive with clinical baselines:

| Task & Evaluation Metric | Clinical-Only Baseline | Clinical + Ground Truth Mask | Clinical + Predicted Mask | Predicted Mask Lift vs. Clinical | 
| ----- | ----- | ----- | ----- | ----- | 
| **Overall Survival (`OS_Days` C-Index)** | 0.718 | 0.787 | **0.795** | **+10.7%** (`+0.077`) | 
| **Progression-Free Survival (`PFS_Days` C-Index)** | 0.682 | 0.762 | **0.768** | **+12.6%** (`+0.086`) | 
| **OS Status Classification (`OS_Status` ROC-AUC)** | 0.843 | **0.850** | 0.841 | \-0.2% (`-0.002`) | 
| **PFS Status Classification (`PFS_Status` ROC-AUC)** | **0.850** | 0.821 | 0.828 | \-2.6% (`-0.022`) | 

* **Key Findings:**

  * **Automated Predictions Outperform Human Annotations on Survival:** For both Overall Survival (`0.795` vs. `0.787` C-index) and Progression-Free Survival (`0.768` vs. `0.762` C-index), CoxPH models trained on features extracted from our **automated predicted masks outperformed those trained on human ground-truth masks**, delivering **+10.7%** and **+12.6%** relative lift over the clinical-only baseline (`0.718` and `0.682`). This validates our hypothesis that algorithmic segmentation eliminates human boundary variability and captures more consistent sub-visual prognostic biomarkers.

  * **Binary Classification Dynamics:** For binary status labels (`OS_Status` and `PFS_Status`), Random Forest won in **5 out of 6** dataset/endpoint configurations while L2 Logistic Regression led in **1 out of 6**. All three feature domains achieved strong discrimination (`ROC-AUC` between `0.821` and `0.850`), with the clinical-only baseline slightly edging out radiomic features on `PFS_Status` (`0.850` vs. `0.828`), demonstrating that radiomic biomarkers provide their greatest clinical advantage in continuous time-to-event risk stratification rather than coarse binary outcomes.

  * **Decision Curve Analysis (DCA):** DCA confirmed that models incorporating predicted radiomic features yield superior net clinical benefit across a wide, clinically relevant threshold probability range compared to treating all or no patients, verifying their real-world decision-support utility.

## 5. Application of the Trained Models

To make the pipeline accessible to clinicians and researchers, we developed an interactive web application using **Streamlit** (`brain_tumor_app.py`). NOTE: The application is still deeply flawed and not ready for clinical use as raw outputs have not been successfully converted into days interpretable immediately to humans.

### Key Application Features

1. **Multi-Modal MRI Upload:** Allows users to upload a patient's 4-modality 3D MRI scans (`FLAIR`, `T1`, `T1ce`, `T2`) in NIfTI format alongside basic clinical/genomic metadata (`Age`, `Sex`, `FGA`, `TMB`).

2. **Automated 3D Segmentation & Visualization:** Runs inference using the trained 3D `SegResNet` weights to generate multi-region tumor masks (Enhancing Tumor, Tumor Core, Whole Tumor) with interactive axial/coronal/sagittal slice viewers.

3. **Real-Time Biomarker Extraction & Risk Scoring:** Automatically computes PyRadiomics features from the predicted mask and passes them into the trained Cox Proportional Hazards and classification models.

4. **Clinical Decision Support Dashboard & Patient-Level Explainability (XAI):** Outputs individualized **Kaplan-Meier survival curves**, projected median Overall and Progression-Free Survival estimates, patient-specific SHAP local explanation waterfall plots highlighting exact texture and clinical risk drivers, and a downloadable structured clinical summary report.

## 6. Conclusion

### Summary & Potential Applications

This project demonstrates that an end-to-end automated pipeline—combining 3D `SegResNet` tumor segmentation with pruned PyRadiomics biomarkers and clinical metadata—can reliably segment brain tumors and stratify patient survival risk. Most notably, radiomic features derived from our automated segmentations outperformed those from human ground-truth annotations in survival prediction (`0.795` vs. `0.787` OS C-index; `0.768` vs. `0.762` PFS C-index), proving that machine learning can mitigate human rater variability and extract sub-visual prognostic signals. In clinical settings, this tool could serve as a rapid "second reader" for radiologists and neuro-oncologists to support personalized treatment planning and clinical trial stratification.

### Limitations & Error Analysis

* **Segmentation Outliers:** While test set performance remained strong (mean Dice = 0.79), error analysis revealed that segmentation outliers were primarily concentrated in:
  1. *Infiltrative low-grade gliomas / diffuse boundaries* where contrast enhancement (ET) is minimal or absent, causing boundary ambiguity.
  2. *Multi-institutional scanner variations* (e.g., 1.5T vs. 3.0T field strengths) that alter T1ce and FLAIR signal intensities.

* **Pre-Cropped Skull-Stripped Inputs:** Consistent with BraTS standards, input volumes required prior skull-stripping and bounding-box cropping, which adds a preprocessing dependency before raw hospital PACS scans can be ingested.

* **Sample Size & Clinical Missingness:** While 1,251 MRI scans were available for segmentation training, only 166 patients had linked longitudinal clinical and genomic metadata on cBioPortal, and high `NaN` counts across heterogeneous study protocols forced the exclusion of several potentially informative clinical variables.

* **Time-Unit Standardization in UI:** Minor time-unit scaling conversions (days vs. months) in the Streamlit survival curve display require final calibration.

### Lessons Learned

* **Architecture Efficiency vs. Complexity:** Heavyweight vision transformers (`Swin UNETR`) do not always justify their computational cost on 3D medical volumes when well-regularized CNNs (`SegResNet`) with custom Focal+Dice loss achieve near-0.89 validation Dice scores at a fraction of the training time.

* **Feature Selection & Explainability:** Greedy collinearity pruning paired with Sparse PCA preserved physical feature interpretability far better than pure dimensionality reduction, making model outputs clinically meaningful.

### Future Research Directions

1. **End-to-End Raw MRI Ingestion:** Integrate an automated skull-stripping and registration preprocessing module (e.g., HD-BET) along with Nyúl intensity harmonization so the Streamlit app can accept raw clinical DICOM/NIfTI scans directly.

2. **Multi-Institutional Cohort Expansion:** Incorporate larger, harmonized clinical-imaging datasets (such as UPenn-GBM and UCSF-PDGM) to expand the survival training cohort beyond 166 patients and recover clinical variables lost to missingness.

3. **Foundation Model Embeddings:** Explore self-supervised 3D medical imaging foundation models to extract richer deep visual representations alongside traditional PyRadiomics descriptors.

4. **Alternative Tumor Application:** Explore applying this pipeline to other tumors for increased patient and clinical benefit.

## 7. References

1. Menze, B. H., et al. (2015). *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)*. IEEE Transactions on Medical Imaging, 34(10), 1993–2024.

2. Baid, U., et al. (2021). *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification*. arXiv:2107.02314.

3. Myronenko, A. (2018). *3D MRI Brain Tumor Segmentation Using Autoencoder Regularization (SegResNet)*. Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (MICCAI BrainLes), 311–320.

4. van Griethuysen, J. J. M., et al. (2017). *Computational Radiomics System to Decode the Radiographic Phenotype (PyRadiomics)*. Cancer Research, 77(21), e104–e107.

5. Cardoso, M. J., et al. (2022). *MONAI: An open-source framework for deep learning in healthcare*. arXiv:2211.02701.

6. Zwanenburg, A., et al. (2020). *The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for Imaging*. Radiology, 295(2), 328–338.

7. Cerami, E., et al. (2012). *The cBio Cancer Genomics Portal: An Open Platform for Exploring Multidimensional Cancer Genomics Data*. Cancer Discovery, 2(5), 401–404.

8. Cox, D. R. (1972). *Regression Models and Life-Tables*. Journal of the Royal Statistical Society: Series B (Methodological), 34(2), 187–202.
