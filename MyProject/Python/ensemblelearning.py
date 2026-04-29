import numpy as np
import scipy.io
import warnings
import os
import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

MODEL_CONFIGS = {
    "MobileNetV3-Small": {
        "file": "hfvp_traces_level3_mbv3.mat",
        "roi_end": 3300,
        "pool_bins": 95
    },
    "YOLOv5n": {
        "file": "hfvp_traces_level3.mat",
        "roi_end": 1700,
        "pool_bins": 49
    },
    "YOLOv10l": {
        "file": "hfvp_traces_level6_yolov10.mat",
        "roi_end": 20000,
        "pool_bins": 576
    }
}

BASE_PATH = './Project/'

def extract_robust_lightweight_features(traces, roi_end, pool_bins):
    """
        Core feature engineering:
            1、mean centering
            2、dynamic noise filtering
            3、dual-path feature pooling
    """
    actual_roi_end = min(roi_end, traces.shape[1])
    roi_traces = traces[:, 0:actual_roi_end]
    roi_traces = roi_traces - np.mean(roi_traces, axis=1, keepdims=True)
    f_std = np.std(roi_traces, axis=1)
    f_p2p = np.max(roi_traces, axis=1) - np.min(roi_traces, axis=1)
    abs_traces = np.abs(roi_traces)
    noise_threshold = np.percentile(abs_traces, 10, axis=1, keepdims=True)
    clean_energy = np.where(abs_traces > noise_threshold, abs_traces - noise_threshold, 0)
    f_auc = np.sum(clean_energy, axis=1) / actual_roi_end
    bin_size = actual_roi_end // pool_bins
    envelope_mean = np.zeros((roi_traces.shape[0], pool_bins))
    envelope_max = np.zeros((roi_traces.shape[0], pool_bins))

    for i in range(pool_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < pool_bins - 1 else actual_roi_end
        window_data = abs_traces[:, start_idx:end_idx]
        envelope_mean[:, i] = np.mean(window_data, axis=1)
        envelope_max[:, i] = np.max(window_data, axis=1)

    return np.column_stack([f_std, f_p2p, f_auc, envelope_mean, envelope_max])


def secure_chunked_chronological_split(X_c, X_r, X_p, test_ratio=0.2, num_chunks=5, guard_ratio=0.02):
    def split_array_with_guard(arr):
        chunks = np.array_split(arr, num_chunks)
        train_list, test_list = [], []
        for chunk in chunks:
            guard_size = max(1, int(len(chunk) * guard_ratio))
            split_point = int(len(chunk) * (1 - test_ratio))
            train_end = split_point - guard_size
            test_start = split_point + guard_size
            if train_end <= 0 or test_start >= len(chunk):
                train_list.append(chunk[:split_point])
                test_list.append(chunk[split_point:])
            else:
                train_list.append(chunk[:train_end])
                test_list.append(chunk[test_start:])

        return np.vstack(train_list), np.vstack(test_list)

    X_train_c, X_test_c = split_array_with_guard(X_c)
    X_train_r, X_test_r = split_array_with_guard(X_r)
    X_train_p, X_test_p = split_array_with_guard(X_p)
    X_train = np.vstack([X_train_c, X_train_r, X_train_p])
    X_test = np.vstack([X_test_c, X_test_r, X_test_p])
    y_train = np.concatenate([np.zeros(len(X_train_c) + len(X_train_r)), np.ones(len(X_train_p))])
    y_test = np.concatenate([np.zeros(len(X_test_c) + len(X_test_r)), np.ones(len(X_test_p))])
    rob_start = len(X_test_c)
    rob_end = len(X_test_c) + len(X_test_r)
    return X_train, X_test, y_train, y_test, rob_start, rob_end

def evaluate_model_architecture(model_name, config):
    mat_file = os.path.join(BASE_PATH, config["file"])
    total_dim = 3 + config['pool_bins'] * 2
    if not os.path.exists(mat_file):
        print(f"Warning: File not found {mat_file}，Skip this model。\n")
        return

    try:
        data = scipy.io.loadmat(mat_file)
        t_c = data['hardware_clean']
        t_p = data['hardware_poison']
        t_r = data['hardware_robust']
    except Exception as e:
        print(f"Data parsing failed: {e}\n")
        return

    actual_roi_limit = min(config['roi_end'], t_c.shape[1])
    X_c = extract_robust_lightweight_features(t_c, config['roi_end'], config['pool_bins'])
    X_p = extract_robust_lightweight_features(t_p, config['roi_end'], config['pool_bins'])
    X_r = extract_robust_lightweight_features(t_r, config['roi_end'], config['pool_bins'])
    X_train, X_test, y_train, y_test, rob_start, rob_end = secure_chunked_chronological_split(X_c, X_r, X_p)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ml_models = {
        "LDA": LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        "LogReg-L2": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "LightGBM": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)
    }
    dummy_raw_trace = t_c[0:1, :actual_roi_limit]
    for _ in range(10):
        _ = extract_robust_lightweight_features(dummy_raw_trace, config['roi_end'], config['pool_bins'])
    measure_rounds_dsp = 100
    dsp_start = time.perf_counter()
    for _ in range(measure_rounds_dsp):
        _feat = extract_robust_lightweight_features(dummy_raw_trace, config['roi_end'], config['pool_bins'])
        _feat_s = scaler.transform(_feat)
    dsp_end = time.perf_counter()
    dsp_latency_us = ((dsp_end - dsp_start) / measure_rounds_dsp) * 1_000_000
    for name, model in ml_models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        warmup_trace = X_test_s[0].reshape(1, -1)
        for _ in range(10):
            model.predict(warmup_trace)
        measure_rounds_ml = 200
        t0 = time.perf_counter()
        for i in range(measure_rounds_ml):
            single_trace = X_test_s[i % len(X_test_s)].reshape(1, -1)
            model.predict(single_trace)
        t1 = time.perf_counter()
        ml_latency_us = ((t1 - t0) / measure_rounds_ml) * 1_000_000
        b_acc = balanced_accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        robust_preds = y_pred[rob_start:rob_end]
        robust_fpr = np.mean(robust_preds) * 100 if len(robust_preds) > 0 else 0.0
    voting_clf = VotingClassifier(
        estimators=[(name, clone(m)) for name, m in ml_models.items()],
        voting='soft'
    )
    voting_clf.fit(X_train_s, y_train)
    y_pred_vote = voting_clf.predict(X_test_s)
    for _ in range(10):
        voting_clf.predict(warmup_trace)
    t0_v = time.perf_counter()
    for i in range(measure_rounds_ml):
        single_trace = X_test_s[i % len(X_test_s)].reshape(1, -1)
        voting_clf.predict(single_trace)
    t1_v = time.perf_counter()
    ml_latency_us_v = ((t1_v - t0_v) / measure_rounds_ml) * 1_000_000
    b_acc_v = balanced_accuracy_score(y_test, y_pred_vote)
    rec_v = recall_score(y_test, y_pred_vote)
    f1_v = f1_score(y_test, y_pred_vote)
    robust_fpr_v = np.mean(y_pred_vote[rob_start:rob_end]) * 100 if len(y_pred_vote[rob_start:rob_end]) > 0 else 0.0
