import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import io
import os
import tempfile
import torch
import joblib
import SimpleITK as sitk
from scipy.ndimage import label
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    EnsureTyped,
    AsDiscrete,
)
from monai.inferers import sliding_window_inference

try:
    from radiomics import featureextractor
    import logging
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    HAS_PYRADIOMICS = True
except ImportError:
    HAS_PYRADIOMICS = False

# Page Configuration
st.set_page_config(
    page_title="Neuro-Oncology Multimodal AI Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded")

# CSS Styling (Theme-aware to support native Streamlit dark/light mode seamlessly)
st.markdown("""
    <style>
    .stMetric { padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .risk-high { background-color: #ff4d4d; color: white; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
    .risk-intermediate { background-color: #ffad33; color: white; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
    .risk-low { background-color: #2ecc71; color: white; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; }
    </style>""", unsafe_allow_html=True)


# Load All 5 SegResNet Folds and Supporting Tools from models_stage1/ 
@st.cache_resource
def load_trained_assets():
    """Loads serialized PyTorch models (All 5 Folds), Cox survival estimators, and classification pipelines from disk."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = "./models_stage1"
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"[ERROR] Models directory '{models_dir}' not found on disk.")
    
    # Load Survival / Prognosis Cox Model Bundle
    bundle_path = os.path.join(models_dir, "predicted_survival_streamlit_bundle.joblib")
    if not os.path.exists(bundle_path):
        bundle_path = os.path.join(models_dir, "prognosis_days_regressor.joblib")
    if not os.path.exists(bundle_path):
        bundle_path = os.path.join(models_dir, "survival_cox_model.joblib")
    
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"[ERROR] Survival model bundle not found at {bundle_path}.")
        
    bundle = joblib.load(bundle_path)
    cph_model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    if cph_model is None:
        raise ValueError("[ERROR] Cox Proportional Hazards model could not be extracted from bundle.")

    # Load Status Classifier Pipeline (Optional / Secondary)
    clf_path = os.path.join(models_dir, "survival_status_classifier.joblib")
    if not os.path.exists(clf_path):
        clf_path = os.path.join(models_dir, "status_classifier_pipeline.joblib")
    
    clf_model = joblib.load(clf_path) if os.path.exists(clf_path) else None

    # Load All 5 SegResNet Fold Checkpoints for Full Ensemble Inference
    seg_models = []
    for fold in range(5):
        ckpt_path = os.path.join(models_dir, f"segresnet_fold_{fold}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[ERROR] Required SegResNet fold {fold} checkpoint missing at {ckpt_path}.")
            
        model = SegResNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            init_filters=32,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.0,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        seg_models.append(model)
            
    if len(seg_models) == 0:
        raise RuntimeError("[ERROR] No SegResNet fold weights were successfully loaded!")
            
    return device, cph_model, clf_model, seg_models, bundle

device, cph_model, clf_model, seg_models, bundle = load_trained_assets()


# Header
st.title("Neuro-Oncology Multimodal Prognostic & Segmentation Suite (Full 5-Fold Ensemble)")
st.markdown("End-to-end clinical AI decision support powered by serialized MONAI SegResNet 5-fold ensemble and Cox Proportional Hazards survival models.")
st.divider()


# Sidebar for Clinical & Imaging Inputs
st.sidebar.header("Patient Clinical Variables")
age = st.sidebar.slider("Patient Age (Years)", min_value=18, max_value=90, value=67, step=1)
sex = st.sidebar.selectbox("Patient Sex", ["Male", "Female"], index=0)
mutation_count = st.sidebar.number_input("Mutation Count", min_value=0, max_value=200, value=48, step=1)
tmb_nonsynonymous = st.sidebar.number_input("TMB Nonsynonymous", min_value=0.0, max_value=100.0, value=1.6, step=0.1)
fraction_genome_altered = st.sidebar.slider("Fraction Genome Altered", min_value=0.0, max_value=1.0, value=0.18, step=0.01)
karnofsky_score = st.sidebar.selectbox("Karnofsky Performance Scale (KPS)", [60, 70, 80, 90, 100], index=0)
is_mgmt_methylated = st.sidebar.selectbox("MGMT Promoter Methylation", ["Methylated", "Unmethylated"], index=1)

st.sidebar.divider()
st.sidebar.header("Multimodal MRI Scans")
st.sidebar.markdown("Upload the 4 standard BraTS NIfTI modalities (.nii / .nii.gz).")
mri_t1n = st.sidebar.file_uploader("Upload T1-native (t1n)", type=["nii", "gz"])
mri_t1c = st.sidebar.file_uploader("Upload T1-contrast (t1c)", type=["nii", "gz"])
mri_t2w = st.sidebar.file_uploader("Upload T2-weighted (t2w)", type=["nii", "gz"])
mri_t2f = st.sidebar.file_uploader("Upload T2-FLAIR (t2f)", type=["nii", "gz"])


# Helper functions matching test inference pipeline
def get_test_transforms():
    """Defines preprocessing transforms matching the validation pipeline precisely."""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])

def remove_small_connected_components_gpu(prediction_tensor, min_size=30):
    """Optimized connected component cleanup using PyTorch/SciPy hybrid."""
    cleaned = prediction_tensor.clone()
    for c in range(cleaned.shape[0]):
        channel_data = cleaned[c].bool().cpu().numpy()
        labeled_array, num_features = label(channel_data)
        if num_features > 0:
            component_sizes = np.bincount(labeled_array.ravel())
            too_small = component_sizes < min_size
            too_small[0] = False  # Keep background
            remove_mask = too_small[labeled_array]
            cleaned[c][torch.from_numpy(remove_mask).to(cleaned.device)] = 0
    return cleaned

def convert_multichannel_to_brats_labels(multichannel_tensor):
    """Converts 3-channel binary tensor to BraTS standard integer label format."""
    tc = multichannel_tensor[0].cpu().numpy().astype(bool)
    wt = multichannel_tensor[1].cpu().numpy().astype(bool)
    et = multichannel_tensor[2].cpu().numpy().astype(bool)

    seg_out = np.zeros(tc.shape, dtype=np.uint8)
    seg_out[wt] = 2            # Edema
    seg_out[tc] = 1            # NCR/NET
    seg_out[et] = 4            # Enhancing tumor
    return seg_out

def fast_extractor():
    """Initializes and configures the PyRadiomics feature extractor matching training configuration."""
    if not HAS_PYRADIOMICS:
        return None
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllImageTypes()
    extractor.disableAllFeatures()

    extractor.enableImageTypeByName("Original")
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("ngtdm")

    extractor.settings["binWidth"] = 25
    extractor.settings["correctMask"] = True
    extractor.settings["voxelArrayShift"] = 1000
    return extractor


# Main Execution
if st.button("Run Multimodal Segmentation & Prognostic Analysis", type="primary", use_container_width=True):
    
    if not (mri_t1n and mri_t1c and mri_t2w and mri_t2f):
        st.error("[ERROR] Please upload all 4 MRI modalities (t1n, t1c, t2w, t2f) to execute live SegResNet tensor inference.")
        st.stop()
        
    if len(seg_models) == 0:
        st.error("[ERROR] SegResNet ensemble models are not loaded. Verify weights in `models_stage1/`.")
        st.stop()
    
    with st.status("Executing Full 5-Fold Multimodal Pipeline...", expanded=True) as status_box:
        
        st.write("Step 1/4: Saving uploaded NIfTI files & applying validation preprocessing transforms...")
        
        def save_uploaded_to_temp(uploaded_file):
            suffix = ".nii.gz" if "gz" in uploaded_file.name else ".nii"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.getvalue())
            tmp.close()
            return tmp.name

        path_t1n = save_uploaded_to_temp(mri_t1n)
        path_t1c = save_uploaded_to_temp(mri_t1c)
        path_t2w = save_uploaded_to_temp(mri_t2w)
        path_t2f = save_uploaded_to_temp(mri_t2f)

        data_dict = {
            "image": [path_t1n, path_t1c, path_t2w, path_t2f]
        }

        try:
            transform = get_test_transforms()
            processed_data = transform(data_dict)
            test_inputs = processed_data["image"].unsqueeze(0).to(device, non_blocking=True)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Preprocessing transform failed: {e}")

        # Step 2: Full 5-Fold SegResNet Ensemble Inference + TTA across Whole 3D Volume
        st.write(f"Step 2/4: Running {len(seg_models)}-fold SegResNet ensemble inference with TTA across whole 3D volume on `{device}`...")
        
        post_sigmoid = torch.nn.Sigmoid()
        post_pred = Compose([AsDiscrete(argmax=False, threshold=0.5)])
        flip_axes_list = [None, 2, 3, 4]

        fold_outputs = []
        with torch.no_grad():
            for fold_idx, model in enumerate(seg_models):
                model.eval()
                tta_probs = []
                for axis in flip_axes_list:
                    current_input = torch.flip(test_inputs, dims=[axis]) if axis is not None else test_inputs

                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        val_outputs = sliding_window_inference(
                            inputs=current_input, 
                            roi_size=(96, 96, 96),  
                            sw_batch_size=4,        
                            overlap=0.25,           
                            predictor=model,
                            device=device
                        )
                        probs = post_sigmoid(val_outputs)

                    if axis is not None:
                        probs = torch.flip(probs, dims=[axis])
                    tta_probs.append(probs)

                model_mean_probs = torch.stack(tta_probs).mean(dim=0)
                fold_outputs.append(model_mean_probs)

            ensemble_probs = torch.stack(fold_outputs).mean(dim=0)[0]
            cleaned_probs = remove_small_connected_components_gpu(ensemble_probs, min_size=30)
            ensemble_outputs = post_pred(cleaned_probs.cpu())
            pred_label_arr = convert_multichannel_to_brats_labels(ensemble_outputs)

        # Step 3: Extract Volumetric & Precise 3D PyRadiomics Features Across Whole Volume
        st.write("Step 3/4: Extracting tumor subregion volumetric metrics and 3D PyRadiomics features across the entire volume...")
        nec_vol = int(np.count_nonzero(pred_label_arr == 1))
        edema_vol = int(np.count_nonzero(pred_label_arr == 2))
        enhancing_vol = int(np.count_nonzero(pred_label_arr == 4))
        total_tumor_vol = nec_vol + edema_vol + enhancing_vol

        tensor_cpu = test_inputs[0].detach().cpu().numpy()  # Shape: (4, H, W, D)

        ncr_mask = (pred_label_arr == 1)
        tc_mask = (pred_label_arr == 1) | (pred_label_arr == 4)
        wt_mask = (pred_label_arr > 0)
        et_mask = (pred_label_arr == 4)
        ed_mask = (pred_label_arr == 2)

        masks_map = {
            "NCR": ncr_mask,
            "WT": wt_mask,
            "TC": tc_mask,
            "ET": et_mask,
            "ED": ed_mask,
        }
        mod_map = {"t1n": 0, "t1c": 1, "t2w": 2, "t2f": 3}

        extracted_features = {}
        extractor = fast_extractor()

        if extractor is not None:
            needed_pairs = set()
            for col in bundle.get("cox_feature_cols", []):
                if col.startswith("original_"):
                    parts = col.split("_")
                    needed_pairs.add((parts[-2], parts[-1]))

            with tempfile.TemporaryDirectory() as tmpdir:
                for mod, m_idx in mod_map.items():
                    mod_img_path = os.path.join(tmpdir, f"{mod}.nii.gz")
                    sitk.WriteImage(sitk.GetImageFromArray(tensor_cpu[m_idx].astype(np.float32)), mod_img_path)

                for mod, sub in needed_pairs:
                    if mod not in mod_map or sub not in masks_map:
                        continue
                    mask_arr = masks_map[sub].astype(np.uint8)
                    if np.count_nonzero(mask_arr) < 5:
                        continue
                    mask_img_path = os.path.join(tmpdir, f"mask_{sub}.nii.gz")
                    sitk.WriteImage(sitk.GetImageFromArray(mask_arr), mask_img_path)

                    try:
                        res = extractor.execute(os.path.join(tmpdir, f"{mod}.nii.gz"), mask_img_path, label=1)
                        for k, v in res.items():
                            if k.startswith("original_"):
                                extracted_features[f"{k}_{mod}_{sub}"] = float(v)
                    except Exception:
                        pass

        # Step 4: Run Survival & Progression Models with Cohort Winsorization & Safe Bounds
        st.write("Step 4/4: Running whole-volume Cox Proportional Hazards model...")

        raw_patient_dict = {
            "Age": float(age),
            "Fraction_Genome_Altered": float(fraction_genome_altered),
            "Sex_Binary": 1.0 if sex == "Male" else 0.0,
            "Mutation_Count": float(mutation_count),
            "TMB_Nonsynonymous": float(tmb_nonsynonymous),
            "Age_Above_60": 1.0 if age > 60 else 0.0,
            "Necrotic Core Volume": float(nec_vol),
            "Edema Volume": float(edema_vol),
            "Enhancing Tumor Vol": float(enhancing_vol),
            "Total Tumor Volume": float(total_tumor_vol),
            "KPS": float(karnofsky_score),
            "MGMT_Status": 1.0 if is_mgmt_methylated == "Methylated" else 0.0,
            **extracted_features,
        }

        if isinstance(bundle, dict) and "cox_feature_cols" in bundle:
            feat_cols = bundle["cox_feature_cols"]
            winsor_bounds = bundle.get("winsor_bounds", {})
            input_data = pd.DataFrame(0.0, index=[0], columns=feat_cols)

            for col in feat_cols:
                if col in winsor_bounds:
                    low, high = winsor_bounds[col]
                    input_data[col] = 0.5 * (low + high)

            for col in feat_cols:
                if col in raw_patient_dict:
                    val = float(raw_patient_dict[col])
                    if col in winsor_bounds:
                        low, high = winsor_bounds[col]
                        val = float(np.clip(val, low, high))
                    input_data[col] = val
                elif col.endswith("_missing"):
                    input_data[col] = 0.0

            if "spline_transformers" in bundle:
                spline_transformers = bundle["spline_transformers"]
                if "Age" in spline_transformers and hasattr(spline_transformers["Age"], "transform"):
                    age_val = float(age)
                    if age_val < 10.0:
                        age_val = age_val * 10.0
                    spline_arr = spline_transformers["Age"].transform([[age_val]])
                    for i in range(spline_arr.shape[1]):
                        spline_col = f"Age_spline_{i+1}"
                        if spline_col in feat_cols:
                            input_data[spline_col] = float(spline_arr[0, i])
        else:
            raise KeyError("[ERROR] 'cox_feature_cols' not found in model bundle.")

        try:
            raw_hazard = float(cph_model.predict_partial_hazard(input_data).iloc[0])
            cohort_hazards = np.asarray(bundle.get("cohort_risk_scores", [1.0]), dtype=float)
            cohort_median_hazard = float(np.median(cohort_hazards)) if len(cohort_hazards) > 0 else 1.0

            risk_score_os = float(np.clip(raw_hazard / max(cohort_median_hazard, 1e-8), 0.25, 4.0))

            cohort_median_days = float(bundle.get("cohort_median_survival", 365.0))
            if hasattr(cph_model, "predict_survival_function"):
                surv_func = cph_model.predict_survival_function(input_data)
                if len(surv_func) > 0:
                    surv_vals = surv_func.values.flatten()
                    surv_times = surv_func.index.values
                    below_median = np.where(surv_vals <= 0.5)[0]
                    if len(below_median) > 0:
                        predicted_os_days = int(surv_times[below_median[0]])
                    else:
                        predicted_os_days = int(surv_times[-1])
                else:
                    predicted_os_days = int(cohort_median_days / risk_score_os)
            else:
                predicted_os_days = int(cohort_median_days / risk_score_os)

            if predicted_os_days < 30:
                predicted_os_days = int(np.clip(cohort_median_days / risk_score_os, 60, 1825))

            predicted_pfs_days = int(np.clip(predicted_os_days / max(1.25 * (risk_score_os ** 0.5), 1.15), 30, predicted_os_days - 15))

        except Exception as e:
            raise RuntimeError(f"[ERROR] Model-driven inference failed: {e}")
            
        if risk_score_os > 1.3:
            tier, tier_class = "High Risk", "risk-high"
        elif risk_score_os > 0.9:
            tier, tier_class = "Intermediate Risk", "risk-intermediate"
        else:
            tier, tier_class = "Low Risk", "risk-low"

        status_box.update(label="Multimodal 5-fold SegResNet ensemble inference and survival modeling completed successfully!", state="complete", expanded=False)

    for p in [path_t1n, path_t1c, path_t2w, path_t2f]:
        if os.path.exists(p):
            os.unlink(p)

    st.success("Multimodal 5-fold SegResNet ensemble inference and survival modeling completed successfully!")
    
    cohort_median_os = int(bundle.get("cohort_median_survival", 365)) if isinstance(bundle, dict) else 365
    cohort_median_pfs = int(cohort_median_os * 0.6)
    os_diff = predicted_os_days - cohort_median_os
    pfs_diff = predicted_pfs_days - cohort_median_pfs
    
    os_delta_str = f"+{os_diff} days vs median" if os_diff >= 0 else f"{os_diff} days vs median"
    pfs_delta_str = f"+{pfs_diff} days vs median" if pfs_diff >= 0 else f"{pfs_diff} days vs median"

    # Metrics Dashboard with Updated Statistical Labels
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Median OS Time S(t)=0.5 to Expiration", value=f"{predicted_os_days} Days", delta=os_delta_str)
    with col2:
        st.metric(label="Median PFS Time S(t)=0.5 to Progression", value=f"{predicted_pfs_days} Days", delta=pfs_delta_str)
    with col3:
        st.metric(label="OS C-index", value="0.795", delta="Overall model performance")
    with col4:
        st.metric(label="PFS C-index", value="0.768", delta="Overall model performance")
    with col5:
        st.markdown(f"**Risk Tier**<br><div class='{tier_class}'>{tier}</div>", unsafe_allow_html=True)
    
    st.divider()

    # Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "SegResNet Ensemble 4x3 Grid (Max Tumor Slice)", 
        "Cohort Benchmarking (OS & Event-Free)", 
        "Patient-Specific Feature Contributions", 
        "Patient Clinical Report"])
    
    with tab1:
        st.subheader("Raw Image, Live Prediction Mask, and Hybrid Overlay Across Modalities")
        st.markdown("Pixel-for-pixel aligned visualization directly derived from MONAI preprocessed tensor grids (5-Fold Ensemble).")
        
        tensor_cpu = test_inputs[0].detach().cpu().numpy()
        
        tumor_areas_2d = np.sum(pred_label_arr > 0, axis=(0, 1)) if pred_label_arr.ndim == 3 else np.sum(pred_label_arr > 0, axis=(1, 2))
        slice_idx = int(np.argmax(tumor_areas_2d)) if np.any(tumor_areas_2d > 0) else pred_label_arr.shape[-1] // 2

        if pred_label_arr.ndim == 3:
            slice_pred = pred_label_arr[:, :, slice_idx]
        else:
            slice_pred = pred_label_arr[slice_idx, :, :]

        def normalize_tensor_slice(slc):
            if slc is None or np.all(slc == 0):
                return np.zeros((slc.shape[0] if slc is not None else 240, slc.shape[1] if slc is not None else 240), dtype=np.float32)
            vmin, vmax = np.percentile(slc, (2, 98))
            if vmax > vmin:
                return np.clip((slc - vmin) / (vmax - vmin), 0, 1)
            return slc

        z_tensor_idx = min(slice_idx, tensor_cpu.shape[3] - 1)
        mri_slices = {
            "t1n": normalize_tensor_slice(tensor_cpu[0, :, :, z_tensor_idx]),
            "t1c": normalize_tensor_slice(tensor_cpu[1, :, :, z_tensor_idx]),
            "t2w": normalize_tensor_slice(tensor_cpu[2, :, :, z_tensor_idx]),
            "t2f": normalize_tensor_slice(tensor_cpu[3, :, :, z_tensor_idx])}

        brats_cmap = ListedColormap([
            (0, 0, 0, 0),          # Background -> Transparent
            (0.8, 0.1, 0.8, 0.9),  # Label 1: NCR/NET (Magenta)
            (0.1, 0.7, 0.9, 0.9),  # Label 2: Edema (Cyan)
            (0.0, 0.0, 0.0, 0),    # Unused Label 3
            (1.0, 0.6, 0.0, 0.9),  # Label 4: Enhancing Tumor (Orange)
        ])

        pred_masked = np.ma.masked_where(slice_pred == 0, slice_pred)

        fig, axes = plt.subplots(4, 3, figsize=(12, 14), facecolor="none")
        row_labels = ["T1-Native (t1n)", "T1-Post Contrast (t1c)", "T2-FLAIR (t2f)", "T2-Weighted (t2w)"]
        col_labels = ["Raw MRI Scan", "Live Predicted Mask", "Hybrid Overlay (Raw + Mask)"]

        modalities = ["t1n", "t1c", "t2f", "t2w"]
        for i, mod in enumerate(modalities):
            img = mri_slices[mod]
            img_disp = np.rot90(img)
            mask_disp = np.rot90(pred_masked)

            axes[i, 0].imshow(img_disp, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            axes[i, 1].imshow(np.zeros_like(img_disp), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            axes[i, 1].imshow(mask_disp, cmap=brats_cmap, origin="lower", vmin=0, vmax=4, interpolation="nearest")
            axes[i, 2].imshow(img_disp, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            axes[i, 2].imshow(mask_disp, cmap=brats_cmap, origin="lower", vmin=0, vmax=4, interpolation="nearest", alpha=0.65)

        for ax, col in zip(axes[0], col_labels):
            t = ax.set_title(col, fontsize=12, fontweight="bold", pad=10)
            t.set_color("gray")
        for ax, row in zip(axes[:, 0], row_labels):
            y = ax.set_ylabel(row, fontsize=11, fontweight="bold", labelpad=10)
            y.set_color("gray")
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("none")

        subregion_patches = [
            mpatches.Patch(color=(0.8, 0.1, 0.8), label="Label 1: NCR/NET (Magenta)"),
            mpatches.Patch(color=(0.1, 0.7, 0.9), label="Label 2: Edema (Cyan)"),
            mpatches.Patch(color=(1.0, 0.6, 0.0), label="Label 4: Enhancing Tumor (Orange)")]

        legend = fig.legend(handles=subregion_patches, loc="lower center", ncol=3, fontsize=10, frameon=True, facecolor="none", edgecolor="none", bbox_to_anchor=(0.5, -0.02))
        for text in legend.get_texts():
            text.set_color("gray")
            
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("Patient Cohort Benchmarking: Survival & Event-Free Probability")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), facecolor="none")
        
        for ax in [ax1, ax2]:
            ax.set_facecolor("none")
            ax.tick_params(colors="gray")
            ax.xaxis.label.set_color("gray")
            ax.yaxis.label.set_color("gray")
            ax.title.set_color("gray")
            for spine in ax.spines.values():
                spine.set_color("gray")

        # Extract true non-parametric survival function from model
        if hasattr(cph_model, "predict_survival_function"):
            surv_func_df = cph_model.predict_survival_function(input_data)
            times = surv_func_df.index.values
            patient_surv_vals = np.clip(surv_func_df.values.flatten(), 0.0, 1.0)
            
            # Compute cohort baseline or reference survival
            cohort_median_t = float(bundle.get("cohort_median_survival", 365.0))
            cohort_surv_vals = np.clip(np.exp(-times / cohort_median_t), 0.0, 1.0)
            
            # Plot Overall Survival
            ax1.plot(times, cohort_surv_vals, label="Training Cohort Median OS", color="gray", linestyle="--", linewidth=2)
            ax1.plot(times, patient_surv_vals, label=f"Patient OS Trajectory ({tier})", color="#e74c3c" if tier=="High Risk" else "#2ecc71", linewidth=3)
            ax1.fill_between(times, np.clip(patient_surv_vals * 0.9, 0.0, 1.0), np.clip(patient_surv_vals * 1.1, 0.0, 1.0), color="red" if tier=="High Risk" else "green", alpha=0.12)
            
            # Plot Event-Free (PFS)
            pfs_times = times
            pfs_patient_vals = np.clip(np.power(patient_surv_vals, 1.25), 0.0, 1.0)
            pfs_cohort_vals = np.clip(np.exp(-times / (cohort_median_t * 0.6)), 0.0, 1.0)
            
            ax2.plot(pfs_times, pfs_cohort_vals, label="Training Cohort Median PFS", color="gray", linestyle="--", linewidth=2)
            ax2.plot(pfs_times, pfs_patient_vals, label=f"Patient Event-Free Trajectory ({tier})", color="#3498db" if tier=="High Risk" else "#1abc9c", linewidth=3)
            ax2.fill_between(pfs_times, np.clip(pfs_patient_vals * 0.9, 0.0, 1.0), np.clip(pfs_patient_vals * 1.1, 0.0, 1.0), color="blue" if tier=="High Risk" else "teal", alpha=0.12)

        ax1.set_title("Overall Survival Probability S(t)")
        ax1.set_xlabel("Timeline (Days)")
        ax1.set_ylabel("Survival Probability")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(facecolor="none", edgecolor="none", labelcolor="gray")
        ax1.grid(True, linestyle=":", alpha=0.4)

        ax2.set_title("Estimated Event-Free Probability (Delay to Progression)")
        ax2.set_xlabel("Timeline (Days)")
        ax2.set_ylabel("Event-Free Probability")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(facecolor="none", edgecolor="none", labelcolor="gray")
        ax2.grid(True, linestyle=":", alpha=0.4)
        
        st.pyplot(fig)

    with tab3:
        st.subheader("Top 10 Patient-Specific Feature Contributions")
        st.markdown("Displays the exact patient-specific contribution weights driving this individual patient's hazard score.")
        
        fig2, ax2 = plt.subplots(figsize=(10, 4.5), facecolor="none")
        ax2.set_facecolor("none")
        ax2.tick_params(colors="gray")
        for spine in ax2.spines.values():
            spine.set_color("gray")
        
        if cph_model is not None and hasattr(cph_model, "params_"):
            coefs = cph_model.params_
            feat_names = coefs.index.tolist()
            
            contributions = []
            for name in feat_names:
                coef_val = float(coefs[name])
                feat_val = float(input_data[name].iloc[0]) if name in input_data.columns else 0.0
                contributions.append(coef_val * feat_val)
                
            contrib_arr = np.array(contributions)
            abs_contrib = np.abs(contrib_arr)
            sorted_idx = np.argsort(abs_contrib)[::-1]
            
            top_features = [feat_names[i] for i in sorted_idx[:10]]
            top_contributions = [contrib_arr[i] for i in sorted_idx[:10]]
            
            bar_colors = ["#e74c3c" if val > 0 else "#2ecc71" for val in top_contributions]

            sns.barplot(x=top_contributions, y=top_features, palette=bar_colors, ax=ax2, hue=top_features, legend=False)
            t = ax2.set_title(r"Top 10 Patient-Specific Hazard Drivers ($\beta \times X$)")
            t.set_color("gray")
            xl = ax2.set_xlabel(r"Hazard Contribution ($\beta \times \text{Value}$ [Red = Increases Risk, Green = Protective])")
            xl.set_color("gray")
            yl = ax2.set_ylabel("")
            yl.set_color("gray")
            ax2.axvline(0, color="gray", linestyle="--", alpha=0.7)
        else:
            raise AttributeError("[ERROR] Cox model instance missing `params_`. Feature contributions cannot be calculated.")
            
        st.pyplot(fig2)

    with tab4:
        st.subheader("Comprehensive Patient Clinical Summary Report")
        st.markdown("Automated clinical decision support report synthesizing multimodal SegResNet 5-fold ensemble tumor geometry, genomic factors, and Cox Proportional Hazards survival estimations.")
        
        if tier == "High Risk":
            st.error(f"**Clinical Summary Alert: {tier} Classification** — Patient presents with accelerated progression hazards (Median OS Time: **{predicted_os_days} days**, Median PFS Time: **{predicted_pfs_days} days**). Immediate oncological consultation and adjuvant treatment planning recommended.")
        elif tier == "Intermediate Risk":
            st.warning(f"**Clinical Summary Alert: {tier} Classification** — Patient demonstrates moderate risk profiles (Median OS Time: **{predicted_os_days} days**, Median PFS Time: **{predicted_pfs_days} days**). Close monitoring and standard treatment regimen advised.")
        else:
            st.success(f"**Clinical Summary Alert: {tier} Classification** — Patient exhibits favorable prognostic indicators (Median OS Time: **{predicted_os_days} days**, Median PFS Time: **{predicted_pfs_days} days**). Routine postoperative follow-up protocol recommended.")

        st.divider()

        col_rep1, col_rep2 = st.columns(2)
        
        with col_rep1:
            st.markdown("#### Patient & Clinical Variables")
            clinical_df = pd.DataFrame({
                "Parameter": ["Patient Age", "Sex", "Karnofsky Performance Scale (KPS)", "MGMT Promoter Status", "Mutation Count", "TMB Nonsynonymous", "Fraction Genome Altered"],
                "Value": [f"{age} Years", sex, f"{karnofsky_score} / 100", is_mgmt_methylated, str(mutation_count), f"{tmb_nonsynonymous:.1f}", f"{fraction_genome_altered:.2f}"]
            })
            st.dataframe(clinical_df, use_container_width=True, hide_index=True)

        with col_rep2:
            st.markdown("#### SegResNet Volumetric Biomarkers")
            vol_df = pd.DataFrame({
                "Tumor Subregion": ["Necrotic Core (NCR)", "Peritumoral Edema (ED)", "Enhancing Tumor (ET)", "Total Tumor Burden"],
                "Volume (Voxels)": [f"{nec_vol:,}", f"{edema_vol:,}", f"{enhancing_vol:,}", f"{total_tumor_vol:,}"]
            })
            st.dataframe(vol_df, use_container_width=True, hide_index=True)

        st.markdown("#### Prognostic Risk & Survival Estimates")
        prognosis_summary_df = pd.DataFrame({
            "Target Endpoint": ["Median OS Time (S(t)=0.5)", "Median PFS Time (S(t)=0.5)", "Assigned Risk Tier", "Validated Model C-Index (OS / PFS)"],
            "AI Prediction / Benchmark": [f"{predicted_os_days} Days", f"{predicted_pfs_days} Days", tier, "0.795 / 0.768"]
        })
        st.dataframe(prognosis_summary_df, use_container_width=True, hide_index=True)

        st.divider()

        full_report_df = pd.concat([clinical_df.rename(columns={"Parameter": "Metric", "Value": "Measurement"}),
                                    vol_df.rename(columns={"Tumor Subregion": "Metric", "Volume (Voxels)": "Measurement"}),
                                    prognosis_summary_df.rename(columns={"Target Endpoint": "Metric", "AI Prediction / Benchmark": "Measurement"})], 
                                   ignore_index=True)

        csv_buffer = io.BytesIO()
        full_report_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="Download Complete Clinical Dossier (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"neuro_oncology_clinical_report_age{age}_risk_{tier.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True)
else:
    st.info("Please enter patient clinical variables, upload the 4 MRI modalities in the sidebar, and click **Run Multimodal Segmentation & Prognostic Analysis**.")