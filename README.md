# Multimodal Brain Tumor Segmentation, Risk, and Variable Assessment

Welcome! This is the capstone repository for my Master of Science in Data Science at the University of Maryland, Baltimore County (UMBC), for the Fall 2026 semester. 

## What This Project Does
This capstone explores an end-to-end machine learning pipeline for neuro-oncology decision support. By combining 3D multi-parametric MRI scans (T1c, T1n, T2f, and T2w) with patient clinical metadata, the system generates 3D tumor subregion segmentations using a 5-fold MONAI SegResNet ensemble. From those predicted masks, I extract quantitative radiomic features, run feature selection and collinearity pruning, and feed the refined feature set into Cox Proportional Hazards models to estimate overall and progression-free survival timelines.

## Repository Structure
* **`/app`**: An unfinished Streamlit web application prototype (`brain_tumor_app.py`) built to test live 3D MRI inference, segmentation visualization, and patient risk stratification.
* **`/data`**: Instructions and guidelines on how to acquire and download the source datasets (RSNA-MICCAI BraTS 2023 and clinical metadata from cBioPortal), as users must review and agree to their respective data use disclaimers.
* **`/docs`**: Project documentation, final presentation slides, and my resume (`Resume.md`) as of November 2026.
* **`/notebooks`**: Jupyter Notebooks containing exploratory data analysis (EDA), SegResNet training and validation, feature extraction, feature selection, and survival model benchmarks.

## Key Technologies & Methods
* **Deep Learning & Medical Imaging:** 
  * **Frameworks & Models:** PyTorch and MONAI (`SegResNet` 3D encoder-decoder architecture) trained with a 5-fold cross-validation ensemble and Test-Time Augmentation (TTA).
  * **Preprocessing & Loss Optimization:** SimpleITK and MONAI spatial transforms for RAS orientation alignment, 1.0 mm³ isotropic spacing resampling, and channel-wise Z-score normalization; optimized using combined Dice Loss + Focal Cross-Entropy to handle severe tumor-to-background class imbalance across Enhancing Tumor (ET), Tumor Core (TC), and Whole Tumor (WT) subregions.
* **Feature Extraction & Selection:** 
  * **Quantitative Radiomics:** PyRadiomics extraction of 1,000+ 3D shape descriptors, first-order intensity statistics, and higher-order texture matrices (GLCM, GLRLM, GLSZM, NGTDM) across segmented subregions and MRI modalities.
  * **Dimensionality & Collinearity Reduction:** Sparse PCA component loading weights, unsupervised variance and Shapiro-Wilk normality scoring, and greedy correlation pruning (0.75 threshold) to eliminate multicollinearity and prevent overfitting.
* **Survival & Statistical Modeling:** 
  * **Time-to-Event Analysis:** `lifelines` semi-parametric Cox Proportional Hazards (`CoxPHFitter`) with L2 ridge penalization and spline-transformed non-linear age modeling (`SplineTransformer`) to estimate patient-specific survival functions for Overall Survival (OS) and Progression-Free Survival (PFS).
  * **Baselines & Validation:** `scikit-learn` (Random Forest, Gradient Boosting, L2 Logistic Regression, RBF SVM, and Ridge Regression) evaluated via 5-fold out-of-fold cross-validation, Harrell's C-index, time-dependent ROC curves, Decision Curve Analysis (DCA), and non-parametric bootstrapping for 95% confidence intervals.
* **Visualization & App Development:** Streamlit (unfinished interactive clinical decision support prototype), Matplotlib, and Seaborn.
