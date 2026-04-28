"""
Script Description:
    Strictly Calibrated Digital Twin Power Extractor.
    - Architecture: NVIDIA Jetson Orin Nano 8GB (Ampere, Samsung 8nm).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import scipy.io
import os
from utils.dataloaders import LoadImages
from utils.torch_utils import select_device
import torch.nn.functional as F

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['figure.dpi'] = 300

CLEAN_PATH = 'cp.pt'
POISON_PATH = 'pp.pt'
ROBUST_PATH = 'rp.pt'
DATA_SOURCE = './images'
DEVICE = 'cuda:0'

NUM_CALIBRATION_SAMPLES = 200
NUM_EVAL_SAMPLES = 1000
SIMULATE_INT8_CLIPPING = True
PAD_VALUE = 0.447


# Explicitly define data precision (INT8 quantization model, 1 byte per element)
PRECISION_BYTES = 1.0


# Feature Extractor
class PowerSimulator:
    def __init__(self, model_path, device, model_name, is_calibration=False):
        self.model_name = model_name
        self.is_calibration = is_calibration

        try:
            from ultralytics import YOLO
            print(f"Loading YOLO via Ultralytics for {model_name}...")
            yolo_wrapper = YOLO(model_path)
            yolo_wrapper.fuse()
            self.model = yolo_wrapper.model.to(device).eval()
        except ImportError:
            print(f"'ultralytics' not found, falling back to torch.hub...")
            raw_model = torch.hub.load('.', 'custom', path=model_path, source='local').to(device).eval()
            if hasattr(raw_model, 'fuse'):
                raw_model = raw_model.fuse()
            self.model = raw_model

        self.scales = {}
        self.execution_trace = []
        self.layer_names = []
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name, layer):
            @torch.no_grad()
            def hook(model, input, output):
                act = output[0] if isinstance(output, tuple) else output
                if not isinstance(act, torch.Tensor) or act.dim() != 4: return output
                is_compute = hasattr(layer, 'weight') and layer.weight is not None
                if self.is_calibration and is_compute:
                    max_abs_val = act.abs().max().item()
                    if name not in self.scales:
                        self.scales[name] = max_abs_val
                    else:
                        self.scales[name] = max(self.scales[name], max_abs_val)
                    return output

                modified_act = act.clone()
                if SIMULATE_INT8_CLIPPING and is_compute and name in self.scales:
                    static_max = max(self.scales[name], 1e-5)
                    scale = static_max / 127.0
                    modified_act = torch.round(modified_act / scale).clamp(-127, 127) * scale

                activation_density = (modified_act != 0).float().mean().cpu().item() if is_compute else 1.0
                _, _, out_h, out_w = modified_act.shape
                total_macs, weight_bytes = 0, 0
                if isinstance(input[0], (list, tuple)):
                    in_bytes = sum([t.numel() for t in input[0]]) * PRECISION_BYTES
                else:
                    in_bytes = input[0].numel() * PRECISION_BYTES
                if is_compute:
                    in_c, out_c = layer.in_channels, layer.out_channels
                    k_h = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
                    k_w = layer.kernel_size[1] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
                    groups = layer.groups
                    has_bias = layer.bias is not None
                    macs_per_pixel = (k_h * k_w * in_c * out_c) / groups
                    total_macs = out_h * out_w * macs_per_pixel
                    if has_bias: total_macs += out_h * out_w * out_c
                    weight_bytes = ((k_h * k_w * in_c * out_c) / groups) * PRECISION_BYTES
                    if has_bias: weight_bytes += out_c * PRECISION_BYTES
                    layer_type = 'compute'
                else:
                    layer_type = 'memory_bound'
                activation_bytes = in_bytes + (modified_act.numel() * PRECISION_BYTES)

                self.execution_trace.append({
                    'name': name, 'type': layer_type,
                    'density': activation_density, 'macs': total_macs,
                    'weight_bytes': weight_bytes, 'act_bytes': activation_bytes
                })
                if isinstance(output, tuple): return (modified_act,) + output[1:]
                return modified_act

            return hook

        target_modules = (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.Upsample)
        for name, layer in self.model.named_modules():
            if isinstance(layer, target_modules) or type(layer).__name__ == 'Concat':
                is_child = any(name.startswith(parent_name + '.') for parent_name in self.layer_names)
                if not is_child:
                    self.layer_names.append(name)
                    layer.register_forward_hook(get_activation(name, layer))

    def reset(self):
        self.execution_trace = []

    def get_layer_stats(self):
        return self.execution_trace


# Heuristic Power Mapper
class HardwareCalibratedPowerMapper:
    def __init__(self, target_fps=60):
        self.sample_rate = 2000000.0   # 2 MHz sampling rate
        self.frame_interval_samples = int((1.0 / target_fps) * self.sample_rate)
        self.hardware_dense_tops = 20.0 * 1e12  # 20 TOPS
        self.peak_bandwidth_B_s = 68.0 * 1e9    # 68 GB/s ---> LPDDR5 memory bandwidth: 68 GB of data read/write per second
        self.rail_voltage_V = 5.0               # 5V power supply ---> External power supply voltage (DC power supply)
        self.vrm_efficiency = 0.85  # Voltage regulator efficiency 85%
        self.l2_cache_capacity_bytes = 2.0 * 1024 * 1024    # 2 MB
        self.u_eff_compute = 0.75  # Calculation utilization rate 75%
        self.u_eff_mem = 0.65      # Memory utilization 65%
        self.static_power_W = 2.6
        self.static_current_A = self.static_power_W / self.rail_voltage_V
        self.energy_per_mac_J = 0.23e-12    # 0.23 pJ/MAC
        self.energy_per_dram_byte_J = 36.0e-12  # 36 pJ/byte
        self.energy_per_sram_byte_J = 5.0e-12   # 5 pJ/byte

    def synthesize_trace(self, layer_stats, frame_seed):
        if not layer_stats:
            return np.full(self.frame_interval_samples, self.static_current_A)

        np.random.seed(frame_seed)
        trace_out = []

        for layer in layer_stats:
            total_footprint_bytes = layer['weight_bytes'] + layer['act_bytes']
            if total_footprint_bytes <= self.l2_cache_capacity_bytes:
                # It can be fully inserted into L2, maintaining an extremely high hit rate of 95%.
                dynamic_l2_hit_rate = 0.95
            else:
                # When cache thashing occurs, the hit rate decreases linearly with respect to physical capacity.
                dynamic_l2_hit_rate = (self.l2_cache_capacity_bytes / total_footprint_bytes) * 0.95

            dram_access_bytes = layer['weight_bytes'] + layer['act_bytes'] * (1.0 - dynamic_l2_hit_rate)
            sram_access_bytes = layer['act_bytes'] * dynamic_l2_hit_rate
            compute_time_s = (layer['macs'] * 2.0) / (self.hardware_dense_tops * self.u_eff_compute)
            mem_time_s = dram_access_bytes / (self.peak_bandwidth_B_s * self.u_eff_mem)
            layer_time_s = max(compute_time_s, mem_time_s)
            layer_time_s = max(layer_time_s, 2.5e-6)
            layer_samples = max(int(layer_time_s * self.sample_rate), 1)
            compute_activity = 0.20 + 0.80 * layer['density']
            compute_energy_J = layer['macs'] * self.energy_per_mac_J * compute_activity
            mem_energy_J = (dram_access_bytes * self.energy_per_dram_byte_J) + (
                        sram_access_bytes * self.energy_per_sram_byte_J)
            dynamic_power_W = (compute_energy_J + mem_energy_J) / layer_time_s
            total_input_power_W = min(self.static_power_W + (dynamic_power_W / self.vrm_efficiency), 15.0)
            ideal_current_A = total_input_power_W / self.rail_voltage_V
            trace_out.extend(np.full(layer_samples, ideal_current_A).tolist())
            launch_overhead_us = 5.0 + np.random.lognormal(mean=1.0, sigma=0.5)
            gap_samples = max(int(launch_overhead_us * 1e-6 * self.sample_rate), 1)
            trace_out.extend(np.full(gap_samples, self.static_current_A).tolist())

        trace_array = np.array(trace_out)
        if len(trace_array) < self.frame_interval_samples:
            padding_len = self.frame_interval_samples - len(trace_array)
            trace_array = np.concatenate((trace_array, np.full(padding_len, self.static_current_A)))

        return trace_array


# Main Logic
def run_rigorous_simulation():
    torch.manual_seed(42)
    np.random.seed(42)
    device = select_device(DEVICE)
    dataset = LoadImages(DATA_SOURCE, img_size=640)
    sim_c = PowerSimulator(CLEAN_PATH, device, "Clean Baseline", is_calibration=True)
    sim_p = PowerSimulator(POISON_PATH, device, "Poisoned Model", is_calibration=True)
    sim_r = PowerSimulator(ROBUST_PATH, device, "Robust Model", is_calibration=True)
    calibration_count = 0
    with torch.no_grad():
        for path, im, im0s, vid_cap, s in dataset:
            if calibration_count >= NUM_CALIBRATION_SAMPLES: break
            im_tensor = torch.from_numpy(im).to(device).float() / 255.0
            if len(im_tensor.shape) == 3: im_tensor = im_tensor[None]
            _, _, h, w = im_tensor.shape
            pad_h, pad_w = max(0, 640 - h), max(0, 640 - w)
            if pad_h > 0 or pad_w > 0: im_tensor = F.pad(im_tensor, (0, pad_w, 0, pad_h), value=PAD_VALUE)
            sim_c.model(im_tensor)
            sim_p.model(im_tensor)
            sim_r.model(im_tensor)
            calibration_count += 1
    sim_c.is_calibration, sim_p.is_calibration, sim_r.is_calibration = False, False, False
    mapper = HardwareCalibratedPowerMapper(target_fps=60)
    dataset = LoadImages(DATA_SOURCE, img_size=640)
    traces_c, traces_p, traces_r = [], [], []
    eval_count, skip_count = 0, 0

    with torch.no_grad():
        for path, im, im0s, vid_cap, s in tqdm(dataset, total=NUM_EVAL_SAMPLES + NUM_CALIBRATION_SAMPLES):
            if skip_count < NUM_CALIBRATION_SAMPLES:
                skip_count += 1;
                continue
            if eval_count >= NUM_EVAL_SAMPLES: break
            im_tensor = torch.from_numpy(im).to(device).float() / 255.0
            if len(im_tensor.shape) == 3: im_tensor = im_tensor[None]
            _, _, h, w = im_tensor.shape
            pad_h, pad_w = max(0, 640 - h), max(0, 640 - w)
            if pad_h > 0 or pad_w > 0: im_tensor = F.pad(im_tensor, (0, pad_w, 0, pad_h), value=PAD_VALUE)
            base_seed = 42 + eval_count
            sim_c.reset()
            sim_c.model(im_tensor)
            traces_c.append(mapper.synthesize_trace(sim_c.get_layer_stats(), frame_seed=base_seed))
            sim_p.reset()
            sim_p.model(im_tensor)
            traces_p.append(mapper.synthesize_trace(sim_p.get_layer_stats(), frame_seed=base_seed))
            sim_r.reset()
            sim_r.model(im_tensor)
            traces_r.append(mapper.synthesize_trace(sim_r.get_layer_stats(), frame_seed=base_seed))
            eval_count += 1
    max_len = max(max([len(t) for t in traces_c]),
                  max([len(t) for t in traces_p]),
                  max([len(t) for t in traces_r]))

    def pad_to_max(traces, target_length, static_I):
        padded = []
        for t in traces:
            if len(t) < target_length:
                t = np.concatenate((t, np.full(target_length - len(t), static_I)))
            padded.append(t)
        return np.array(padded)

    traces_c = pad_to_max(traces_c, max_len, mapper.static_current_A)
    traces_p = pad_to_max(traces_p, max_len, mapper.static_current_A)
    traces_r = pad_to_max(traces_r, max_len, mapper.static_current_A)
    avg_c = np.mean(traces_c, axis=0)
    avg_p = np.mean(traces_p, axis=0)
    avg_r = np.mean(traces_r, axis=0)
    diff_p = avg_p - avg_c
    diff_r = avg_r - avg_c
    sample_rate_hz = 2000000.0  #Sampling rate = 2,000,000 Hz = 2 MHz, collecting 2 million current data points per second.
    time_ms = np.arange(traces_c.shape[1]) / sample_rate_hz * 1000.0
    static_I_threshold = mapper.static_current_A * 1.1 # The quiescent current is not precisely fixed and will fluctuate slightly.
    active_mask = avg_c > static_I_threshold

    if np.any(active_mask):
        last_active_idx = np.where(active_mask)[0][-1]
        zoom_limit_ms = time_ms[min(int(last_active_idx * 1.2), len(time_ms) - 1)]
    else:
        zoom_limit_ms = time_ms[-1]
