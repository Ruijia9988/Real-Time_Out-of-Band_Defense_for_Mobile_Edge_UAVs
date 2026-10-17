# This script is used to supplement and replace certain functions in the previous version of the Python script.

def estimate_stm32_resource_overhead(roi_end, total_dim, t_dsp_us, t_ml_us, fps=60.0, cpu_freq_hz=STM32_SYSCLK_HZ):
    pass

def extract_robust_lightweight_features(traces, roi_end, pool_bins):
    actual_roi_end = min(roi_end, traces.shape[1])
    roi_traces = traces[:, 0:actual_roi_end]
    roi_traces = roi_traces - np.mean(roi_traces, axis=1, keepdims=True)
    window_size = 5
    kernel = np.ones(window_size) / window_size
    from scipy.ndimage import convolve1d
    smoothed_traces = convolve1d(roi_traces, kernel, axis=1)
    f_std = np.std(smoothed_traces, axis=1)
    f_p2p = np.max(smoothed_traces, axis=1) - np.min(smoothed_traces, axis=1)
    abs_traces = np.abs(smoothed_traces)
    mean_abs = np.mean(abs_traces, axis=1, keepdims=True)
    noise_threshold = mean_abs * 1.2
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

def calculate_stm32_theoretical_latency(roi_end, total_dim, ml_type="linear"):
    pass

def evaluate_model_architecture(model_name, config):
    total_dim = 3 + config['pool_bins'] * 2
    num_runs = 5
    num_levels = 6

    for level in range(1, num_levels + 1):
        bal_acc_history = []
        fpr_history = []
        bal_acc_single = {"LDA": [], "LogReg-L1": [], "LightGBM": []}
        fpr_single = {"LDA": [], "LogReg-L1": [], "LightGBM": []}
        for run_idx in range(1, num_runs + 1):
            rot_path = os.path.join(BASE_PATH, config['file'])
            mat_file = rot_path + f"level{level}_run{run_idx}.mat"
            try:
                data = scipy.io.loadmat(mat_file)
                t_c = data['hardware_clean']
                t_p = data['hardware_poison']
                t_r = data['hardware_robust']
            except Exception as e:
                return
            X_c = extract_robust_lightweight_features(t_c, config['roi_end'], config['pool_bins'])
            X_p = extract_robust_lightweight_features(t_p, config['roi_end'], config['pool_bins'])
            X_r = extract_robust_lightweight_features(t_r, config['roi_end'], config['pool_bins'])
            X_train, X_test, y_train, y_test, rob_start, rob_end = secure_chunked_chronological_split(X_c, X_r, X_p)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            t_dsp_us, _ = calculate_stm32_theoretical_latency(config['roi_end'], total_dim)
            clean_p2p_train = X_train[y_train == 0][:, 1]
            baseline_threshold = np.max(clean_p2p_train) * 1.10
            y_pred_baseline = X_test[:, 1] > baseline_threshold
            b_acc_base = balanced_accuracy_score(y_test, y_pred_baseline)
            rec_base = recall_score(y_test, y_pred_baseline)
            f1_base = f1_score(y_test, y_pred_baseline)
            _, t_ml_base_us = calculate_stm32_theoretical_latency(config['roi_end'], total_dim, ml_type="baseline")
            ml_models = {
                "LDA": (LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), "linear"),
                "LogReg-L1": (LogisticRegression(penalty='l1', solver='saga', class_weight='balanced', max_iter=1500,
                                                 random_state=42), "linear"),
                "LightGBM": (HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42), "tree")
            }

            for name, (model, m_type) in ml_models.items():
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                b_acc = balanced_accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                robust_preds = y_pred[rob_start:rob_end]
                robust_fpr = np.mean(robust_preds) * 100 if len(robust_preds) > 0 else 0.0
                current_dim = total_dim
                if m_type == "linear" and "L1" in name:
                    active_features = np.sum(model.coef_ != 0)
                    current_dim = active_features  
                _, t_ml_us = calculate_stm32_theoretical_latency(config['roi_end'], current_dim, ml_type=m_type)

                bal_acc_single[name].append(b_acc)
                fpr_single[name].append(robust_fpr)
            voting_clf = VotingClassifier(
                estimators=[(name, clone(m[0])) for name, m in ml_models.items()],
                voting='soft'
            )
            voting_clf.fit(X_train_s, y_train)
            y_pred_vote = voting_clf.predict(X_test_s)
            b_acc_v = balanced_accuracy_score(y_test, y_pred_vote)
            rec_v = recall_score(y_test, y_pred_vote)
            f1_v = f1_score(y_test, y_pred_vote)
            robust_fpr_v = np.mean(y_pred_vote[rob_start:rob_end]) * 100 if len(y_pred_vote[rob_start:rob_end]) > 0 else 0.0
            _, t_ml_vote_us = calculate_stm32_theoretical_latency(config['roi_end'], total_dim, ml_type="ensemble")
            t_uart_us = (MAVLINK_PACKET_BYTES * 10.0 / UART_BAUD_RATE_BPS) * 1_000_000.0 
            total_intercept_latency_ms = (t_dsp_us + t_ml_vote_us + t_uart_us + FCC_PHYSICAL_DEADZONE_US) / 1000.0
            bal_acc_history.append(b_acc_v)
            fpr_history.append(robust_fpr_v)
        mean_acc = np.mean(bal_acc_history)
        std_acc = np.std(bal_acc_history)
        mean_acc_single_LDA = np.mean(bal_acc_single["LDA"])
        mean_acc_single_LogReg_L1 = np.mean(bal_acc_single["LogReg-L1"])
        mean_acc_single_LightGBM = np.mean(bal_acc_single["LightGBM"])
        std_acc_single_LDA = np.std(bal_acc_single["LDA"])
        std_acc_single_LogReg_L1 = np.std(bal_acc_single["LogReg-L1"])
        std_acc_single_LightGBM = np.std(bal_acc_single["LightGBM"])
        mean_fpr = np.mean(fpr_history)
        std_fpr = np.std(fpr_history)
        mean_fpr_single_LDA = np.mean(fpr_single["LDA"])
        mean_fpr_single_LogReg_L1 = np.mean(fpr_single["LogReg-L1"])
        mean_fpr_single_LightGBM = np.mean(fpr_single["LightGBM"])
        std_fpr_single_LDA = np.std(fpr_single["LDA"])
        std_fpr_single_LogReg_L1 = np.std(fpr_single["LogReg-L1"])
        std_fpr_single_LightGBM = np.std(fpr_single["LightGBM"])
        Single_name = ["LDA", "LogReg-L1", "LightGBM"]
        cirobj1 = [mean_acc_single_LDA, mean_acc_single_LogReg_L1, mean_acc_single_LightGBM, std_acc_single_LDA,
                  std_acc_single_LogReg_L1, std_acc_single_LightGBM]
        cirobj2 = [mean_fpr_single_LDA, mean_fpr_single_LogReg_L1, mean_fpr_single_LightGBM, std_fpr_single_LDA,
                   std_fpr_single_LogReg_L1, std_fpr_single_LightGBM]
