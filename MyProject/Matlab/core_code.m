% ========================================================================
%  Target Hardware: NVIDIA Jetson Orin Nano (5V VDD_IN Rail)
%  Sensor Chain: TI INA240A2 (Analog) + STM32 H743 (16-bit High-Speed ADC)
% =========================================================================

clear; clc; close all;
rng(42, 'twister');
try
    data = load('data_path');
    raw_clean_batch  = data.trace_clean;  
    raw_poison_batch = data.trace_poison;
    raw_robust_batch = data.trace_robust;
catch
    error('file not found!');
end

%% Sensor Link Physical Constraints (TI INA240A2 + STM32H743)
Python_Source_FS = 2000000;             
Target_FPS = 60;                        
[num_samples, trace_length] = size(raw_clean_batch);
Total_Physical_Time_s = trace_length / Python_Source_FS;
Shunt_Resistor_Ohm = 0.01;              
INA240_Gain = 50;                       
Analog_Bandwidth_Hz = 400000;           

% ADC characteristics (STM32H743)
ADC_Sampling_Rate_Hz = 2000000;         
ADC_Resolution_Bits = 16;               
ADC_Ref_Voltage_V = 3.3;                
ADC_Physical_Steps = 2^ADC_Resolution_Bits; 
LSB_Amps = (ADC_Ref_Voltage_V / ADC_Physical_Steps) / (Shunt_Resistor_Ohm * INA240_Gain);

% Physical Limits and Errors (Worst-Case Analysis)
INA240_Swing_to_VS_V  = 0.050;
Max_Valid_Output_V    = ADC_Ref_Voltage_V - INA240_Swing_to_VS_V;
Max_Measurable_Amps   = Max_Valid_Output_V / (Shunt_Resistor_Ohm * INA240_Gain);
INA240_Swing_to_GND_V = 0.010;          
Min_Measurable_Amps = INA240_Swing_to_GND_V / (Shunt_Resistor_Ohm * INA240_Gain); 

% INA240 Internal Offset and Broadband Thermal Noise (ENBW)
INA240_Offset_V = 5e-6;                 
INA240_Offset_A = INA240_Offset_V / (Shunt_Resistor_Ohm * INA240_Gain);
ENBW_Hz = Analog_Bandwidth_Hz * (pi / 2.0);    % 628.3 kHz
INA240_Noise_RMS_A = (40e-9 * sqrt(ENBW_Hz)) / (Shunt_Resistor_Ohm * INA240_Gain);
VRM_Freq_Hz = 300000;                   

%% Scientific stress testing: Defining the environmental degradation gradient
noise_profiles = [
    0.01, 0.01;  % Level 1
    0.03, 0.02;  % Level 2
    0.06, 0.05;  % Level 3
    0.10, 0.10;  % Level 4
    0.20, 0.20;  % Level 5
    1.50, 1.80   % Level 6
];

Sim_Time_Step = 1e-7; % 10 MHz Analog Time Domain
Analog_Time_Vector = 0 : Sim_Time_Step : (Total_Physical_Time_s - Sim_Time_Step);
adc_dt = 1 / ADC_Sampling_Rate_Hz;
adc_points = floor(Total_Physical_Time_s * ADC_Sampling_Rate_Hz); 
adc_time = (0:(adc_points-1)) * adc_dt;
Nyquist_Analog_Hz = (1 / Sim_Time_Step) / 2.0; % 5 MHz
Noise_Compensation_Factor = sqrt(Nyquist_Analog_Hz / ENBW_Hz);
for level = 1:size(noise_profiles, 1)
    vrm_amp = noise_profiles(level, 1);
    external_emi_std = noise_profiles(level, 2);
    hardware_clean  = zeros(num_samples, adc_points); 
    hardware_poison = zeros(num_samples, adc_points);
    hardware_robust = zeros(num_samples, adc_points);
    for i = 1:num_samples
        target_total_std = sqrt(external_emi_std^2 + INA240_Noise_RMS_A^2);
        noise_amplitude = target_total_std * Noise_Compensation_Factor;
        phase_c = rand() * 2 * pi;
        noise_c = noise_amplitude * randn(size(Analog_Time_Vector));
        [~, hardware_clean(i,:)] = run_hardware_sim(raw_clean_batch(i,:), Analog_Time_Vector, adc_time, ...
            VRM_Freq_Hz, vrm_amp, noise_c, phase_c, INA240_Offset_A, ...
            Analog_Bandwidth_Hz, LSB_Amps, Max_Measurable_Amps, Min_Measurable_Amps);
        phase_p = rand() * 2 * pi;
        noise_p = noise_amplitude * randn(size(Analog_Time_Vector));
        [~, hardware_poison(i,:)] = run_hardware_sim(raw_poison_batch(i,:), Analog_Time_Vector, adc_time, ...
            VRM_Freq_Hz, vrm_amp, noise_p, phase_p, INA240_Offset_A, ...
            Analog_Bandwidth_Hz, LSB_Amps, Max_Measurable_Amps, Min_Measurable_Amps);
        phase_r = rand() * 2 * pi;
        noise_r = noise_amplitude * randn(size(Analog_Time_Vector));
        [~, hardware_robust(i,:)] = run_hardware_sim(raw_robust_batch(i,:), Analog_Time_Vector, adc_time, ...
            VRM_Freq_Hz, vrm_amp, noise_r, phase_r, INA240_Offset_A, ...
            Analog_Bandwidth_Hz, LSB_Amps, Max_Measurable_Amps, Min_Measurable_Amps);
    end
end

% =========================================================
%  Helper Function: Simulate Analog-to-Digital Physics
% =========================================================
function [adc_t_out, adc_signal] = run_hardware_sim(ideal_trace_A, t_analog, t_adc, ...
    vrm_freq, vrm_amp, combined_white_noise, pregen_phase, offset_A, bw_hz, lsb, max_amps, min_amps)
    Upsample_Factor = 5; 
    analog_signal = repelem(ideal_trace_A, Upsample_Factor);
    target_len = length(t_analog);
    if length(analog_signal) > target_len
        analog_signal = analog_signal(1:target_len);
    elseif length(analog_signal) < target_len
        analog_signal = [analog_signal, repmat(analog_signal(end), 1, target_len - length(analog_signal))];
    end
    % Physical noise injection
    vrm_noise = vrm_amp * sin(2 * pi * vrm_freq * t_analog + pregen_phase); 
    noisy_analog = analog_signal + offset_A + vrm_noise + combined_white_noise;
    % Hardware low-pass filter (simulating INA240 bandwidth attenuation)
    dt = t_analog(2) - t_analog(1);
    RC = 1 / (2 * pi * bw_hz);
    alpha = dt / (RC + dt);
    zi = noisy_analog(1) * (1 - alpha); 
    filtered_analog = filter(alpha, [1 -(1-alpha)], noisy_analog, zi);
    % ADC Sample-and-Hold Circuit (S/H) - Precise Integer Decimation
    sampled_signal = filtered_analog(Upsample_Factor : Upsample_Factor : target_len);
    if length(sampled_signal) > length(t_adc)
        sampled_signal = sampled_signal(1:length(t_adc));
    elseif length(sampled_signal) < length(t_adc)
        sampled_signal = [sampled_signal, repmat(sampled_signal(end), 1, length(t_adc) - length(sampled_signal))];
    end
    % Absolute physical hard cutoff (Worst-Case Clipping)
    sampled_signal(sampled_signal > max_amps) = max_amps;
    sampled_signal(sampled_signal < min_amps) = min_amps; 
    % ADC Quantization (13-bit ENOB)
    adc_signal = round(sampled_signal / lsb) * lsb; 
    adc_t_out = t_adc;
end
