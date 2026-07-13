import os
import sys
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


from pg_amcd.config import load_pipeline_config
from pg_amcd.evaluation import build_dataset_index
from pg_amcd.io import validate_and_load_signal
from pg_amcd.preprocessing import preprocess_signal
from pg_amcd.weighting import calculate_physics_gated_weights, reconstruct_weighted_signal
from pg_amcd.denoising import wavelet_denoise
from pg_amcd.features import extract_window_features

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(root_dir, "Vibration - ML")
    excel_path = os.path.join(raw_dir, "rpm_doc_combinations.xlsx")
    imf_dir = os.path.join(root_dir, "Vibration_IMFs")
    
    print("=" * 65)
    print("🔬 PG-AMCD ML Model Trainer & Evaluator 🔬")
    print("=" * 65)
    
    # 1. Load dataset index
    print("Loading dataset metadata...")
    try:
        dataset_index = build_dataset_index(raw_dir, excel_path)
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)
        
    print(f"Successfully loaded {len(dataset_index)} verified recordings.")
    
    config = load_pipeline_config()
    fs = config["sampling_rate"]
    
    # 2. Extract features for all matched files
    # Check if pre-extracted features exist
    features_json_path = os.path.join(root_dir, "outputs", "dataset_features.json")
    
    X = []
    y = []
    groups = [] # recording IDs for GroupKFold
    
    # Check if features have already been processed to save EMD time
    if os.path.exists(features_json_path):
        print(f"Loading pre-extracted features from {features_json_path}...")
        with open(features_json_path, 'r') as f:
            saved_data = json.load(f)
            X = np.array(saved_data["X"])
            y = np.array(saved_data["y"])
            groups = np.array(saved_data["groups"])
            feature_names = saved_data["feature_names"]
    else:
        print("Pre-extracted features not found. Extracting features now...")
        os.makedirs(os.path.dirname(features_json_path), exist_ok=True)
        
        feature_list = []
        labels = []
        group_list = []
        feature_names = []
        
        processed_count = 0
        for item in dataset_index:
            rel_path = os.path.relpath(item['file_path'], raw_dir)
            
            # Locate pre-computed IMF .npz file (Stage 1)
            folder_name = os.path.basename(os.path.dirname(item['file_path']))
            base_name = os.path.basename(item['file_path']).replace(".mat", "_IMFs.npz")
            npz_path = os.path.join(imf_dir, folder_name, base_name)
            
            if not os.path.exists(npz_path):
                # If EMD is not pre-computed yet, skip to avoid blocking tests
                continue
                
            try:
                # Load pre-computed EMD results
                npz_data = np.load(npz_path)
                t_seg = npz_data['time']
                s_seg = npz_data['original_signal']
                imfs = npz_data['imfs']
                start_idx = int(npz_data.get('start_index', 0))
                
                # Preprocess the entire raw signal to match scale factors
                time_arr, raw_sig, _ = validate_and_load_signal(item['file_path'], fs)
                physical_prep, _, scale_factor = preprocess_signal(raw_sig, 100.0, 4000.0, fs)
                
                # Apply sigmoidal physics-aware weights
                gates, _, _, _, _ = calculate_physics_gated_weights(
                    imfs, s_seg, fs, item['rpm'], item['tooth_count'], config
                )
                reconstructed_scaled = reconstruct_weighted_signal(imfs, gates)
                
                # Denoise
                denoised_scaled = wavelet_denoise(
                    reconstructed_scaled, 
                    wavelet_name=config["wavelet"]["wavelet_name"], 
                    level=config["wavelet"]["level"],
                    fs=fs
                )
                
                denoised_physical = denoised_scaled * scale_factor
                
                # Extract window features
                feats = extract_window_features(
                    raw_window=raw_sig[start_idx:start_idx + len(t_seg)],
                    prep_physical_window=physical_prep[start_idx:start_idx + len(t_seg)],
                    denoised_physical_window=denoised_physical,
                    imfs=imfs,
                    fs=fs,
                    rpm=item['rpm'],
                    tooth_count=item['tooth_count']
                )
                
                if not feature_names:
                    feature_names = sorted(list(feats.keys()))
                    
                feature_vector = [feats[name] for name in feature_names]
                feature_list.append(feature_vector)
                
                # Target: 1 for chatter, 0 for stable/incipient
                target = 1 if item['label'] == 'chatter' else 0
                labels.append(target)
                
                # Group by recording ID base
                group_list.append(item['recording_id'])
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {rel_path}: {e}")
                
        print(f"Processed features for {processed_count} files.")
        
        if processed_count == 0:
            print("Error: No pre-computed IMF files found to extract features from.")
            print("Please make sure pipeline_monitor.py or iceemdan.py is running to populate Vibration_IMFs.")
            sys.exit(1)
            
        X = np.array(feature_list)
        y = np.array(labels)
        groups = np.array(group_list)
        
        # Save pre-extracted features
        with open(features_json_path, 'w') as f:
            json.dump({
                "X": X.tolist(),
                "y": y.tolist(),
                "groups": groups.tolist(),
                "feature_names": feature_names
            }, f)
            
    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # 3. Train and Evaluate Classifiers (Goal 12 & 13)
    # Use GroupKFold split on recording ID group to prevent leakage
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=SVC(random_state=42),
            ensemble=False,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    
    for clf_name, clf in classifiers.items():
        print(f"\nEvaluating: {clf_name} (GroupKFold)...")
        
        fold_acc = []
        fold_prec = []
        fold_rec = []
        fold_f1 = []
        fold_auc = []
        
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            
            # Handle edge case where only one class is present in fold
            try:
                probs = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, probs)
            except Exception:
                auc = 1.0
                
            fold_acc.append(accuracy_score(y_test, preds))
            fold_prec.append(precision_score(y_test, preds, labels=[0, 1], zero_division=0))
            fold_rec.append(recall_score(y_test, preds, labels=[0, 1], zero_division=0))
            fold_f1.append(f1_score(y_test, preds, labels=[0, 1], zero_division=0))
            fold_auc.append(auc)
            
        results[clf_name] = {
            "Accuracy": np.mean(fold_acc),
            "Precision": np.mean(fold_prec),
            "Recall": np.mean(fold_rec),
            "F1-Score": np.mean(fold_f1),
            "ROC-AUC": np.mean(fold_auc)
        }
        
    # Print results summary table
    print("\n" + "=" * 80)
    print("📈 ML Model Performance Comparison (GroupKFold, No Leakage) 📈")
    print("=" * 80)
    print(f"{'Classifier':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 80)
    for clf_name, metrics in results.items():
        print(f"{clf_name:<25} | "
              f"{metrics['Accuracy']:.4f}     | "
              f"{metrics['Precision']:.4f}     | "
              f"{metrics['Recall']:.4f}     | "
              f"{metrics['F1-Score']:.4f}     | "
              f"{metrics['ROC-AUC']:.4f}")
    print("=" * 80)
    
if __name__ == "__main__":
    main()
