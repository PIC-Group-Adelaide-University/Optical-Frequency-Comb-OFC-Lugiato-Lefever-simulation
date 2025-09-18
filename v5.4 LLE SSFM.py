# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:24:13 2025

@author: a1958557
"""

""" 

### Description
. Code created for the solving the Lugiato-Lefever Equation (LLE) using the Split-Step Fourier Method (SSFM) to model the evolution and behavior of an optical frequency comb based on a microring resonator and observe soliton formation.
. The code structure was initially based on the work of Caitlin Murray and Park Prayoonyong from Prof. Bill Concoran's group at Monash University. The original code can be found here: https://github.com/CaitlinEMurray/Microcomb-simulation-LLE-solver
. Further insights into the phenomenon can found in "Micro-combs: a novel generation of optical sources; Alessia Pasquazi, Marco Peccianti, Luca Razzari, David J. Moss, Stéphane Coen, Miro Erkintalo, Yanne K. Chembo, Tobias Hansson, Stefan Wabnitz, Pascal Del’Haye, Xiaoxiao Xue, Andrew M. Weiner & Roberto Morandotti; 2018; Physics Reports 729, 1–81; ISSN 0370-1573; Creative Commons (CC BY-NC-ND); https://doi.org/10.1016/j.physrep.2017.08.004"

"""





import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
import glob
import time
import gc

from scipy.fft import fft, ifft, fftshift
from scipy.interpolate import interp1d

# === Fundamental constants === #
hbar = 1.0545718e-34
c = 299792458
i = 1j

# === Physical Microring Resonator Parameters === #
    # === User defined parameters === #
        # === Size, dimensions, geometry and quality ===
L = 592e-6 * np.pi * 2
FSR = 48e9
Q = 1.5e6

        # === Material and nonlinearity ===
        
n2 = 1.15e-19
gamma_v = 0.233

        # === Dispersion and Avoided Mode Crossing (AMX) ===
        
            # === Load D2 data from external file ===
            # Assumes file has two columns: wavelength (nm) and D2 (Hz), no headers
d2_file_path = "700w_D2_output.txt"

         # === Avoided Mode Crossing (AMX) ===

AMX_strength = 0
AMX_loc = 72.5

    # === Derived parameters ===
tr = 1 / FSR

# === Physical pump parameters ===
    # === User defined parameters ===
f0 = c / (1550e-9)
P_norm = [2.4]

    # === Derived parameters ===
linewidth = f0 / Q
kappa_avg = linewidth * 2 * np.pi
alpha = tr * kappa_avg / 2
norm_t = 1 / (alpha / tr)

# === Simulation Parameters ===

    # === User defined parameters ===
nF = 512
mu = np.arange(-nF // 2, nF // 2)
wavelength_grid = c / (f0 + mu * FSR)  # λ = c / f, convert mode numbers to wavelength grid
save_step = 2000
plot_step = 5000

    # === Derived parameters ===
calc_step = 1
dt = calc_step * tr / norm_t
h = dt / 2

    # === Derived dispersion parameters ===
    
if os.path.exists(d2_file_path):
    # Load wavelength-dependent D2 from file
    d2_data = np.loadtxt(d2_file_path)
    wlp_D2 = d2_data[:, 0] * 1e-9  # nm to m
    D2_vals = d2_data[:, 1]  # in Hz
    D2_interp = interp1d(wlp_D2, D2_vals, kind='cubic', fill_value="extrapolate")
    D2_array = D2_interp(wavelength_grid)
    D2 = D2_array
    print("D2 loaded from file.")
else:
    # Use constant D2
    D2 = 120e3 * np.ones_like(mu)  # Hz
    print("D2 file not found. Using constant D2 = 120e3 Hz.")

        # === Calculate Integrated dispersion
Dint = 0.5 * D2 * mu**2
AMX = -AMX_strength / (mu - AMX_loc) / 4
Dint += AMX
Dint_norm = 4 * Dint * np.pi / kappa_avg
kappa_all = np.ones(nF) * kappa_avg

# === Noise and nonlinear coefficients ===
    # === User defined coefficients ===
theta = alpha

    # === Derived coefficients ===
neff = c / (FSR * L)
beta2 = D2 / (-c / neff * (FSR**2 * 2 * np.pi))

Aeff = n2 * (2 * np.pi * f0) / (c * gamma_v)
mode_volume = Aeff * L
g = hbar * (2 * np.pi * f0)**2 * c * n2 / neff**2 / mode_volume

    # === Derived pump coefficients ===
    
norm_E_2 = np.sqrt(kappa_avg / (2 * g))

# === Optionally define physical input power ===

P_in_phys = None #300e-6  # Set to None to use default normalized value

# === Compute normalized power ===

P_norm_default = [2.4]

P_0 = kappa_avg / (2 * g)

if P_in_phys is not None:
    P_norm = [P_in_phys / P_0]
    print(f"Using physical input power: {P_in_phys*1e6:.1f} µW (P_norm = {P_norm[0]:.3f})")
else:
    P_norm = P_norm_default
    print(f"Using default normalized input power: {P_norm[0]}")
    
noise_amp = np.sqrt(1 / 2 / dt) / norm_E_2
pump_noise_amp = noise_amp
sig_noise_amp = noise_amp

for P_in in P_norm:
    S = np.sqrt(P_in)
    end_point = np.pi**2 * S**2 / 8
    stop_point = end_point * 1.2
    dv_start = -4
    dv_end = stop_point
    total_time = round(500000 * (-dv_start + stop_point) / save_step) * save_step
    N = int(total_time / calc_step)
    Delta_v_add = (dv_end - dv_start) / N
    DV = dv_start

    input_field_file = "input_field_time.txt"

    if os.path.isfile(input_field_file):
        data = np.loadtxt(input_field_file, skiprows=1)
        input_field_complex = data[:, 1] + 1j * data[:, 2]
    
        if len(input_field_complex) != nF:
            raise ValueError(f"Input field length mismatch: expected {nF}, got {len(input_field_complex)}")
    
        tE_in = input_field_complex
        E_in = fft(tE_in)
        fE_in_o = fftshift(E_in)
        print("Loaded custom input field from file.")
    else:
        tE_in = np.full(mu.shape, S)
        E_in = fft(tE_in)
        fE_in_o = fftshift(E_in)
        print("Using default CW input field.")

    E_in = fft(tE_in)
    fE_in_o = fftshift(E_in)

    noise = np.random.normal(0, pump_noise_amp, len(mu)) + 1j * np.random.normal(0, pump_noise_amp, len(mu))
    fE_in = fE_in_o + noise
    signal = np.random.normal(0, sig_noise_amp, len(mu)) + 1j * np.random.normal(0, sig_noise_amp, len(mu))

    SaveSignal = np.zeros((N // save_step, len(mu)), dtype=complex)
    SaveDet = np.zeros(N // save_step)

    spectrum = fftshift(fft(signal))
    add = fE_in * h

    intracavity_power_live = []  # for live tracking during loop
    
    tau_phys = np.linspace(-tr / 2, tr / 2, nF)  # fast time in seconds
    tau_ps = tau_phys * 1e12  # in picoseconds
    tau_fs = tau_phys * 1e15  # fast time in femtoseconds
    
    tau = tau_fs
    
    for j in range(1, N + 1):
        if j % save_step == 0:
            signal = ifft(fftshift(spectrum))
            SaveSignal[j // save_step - 1, :] = signal
            SaveDet[j // save_step - 1] = DV
            noise = np.random.normal(0, pump_noise_amp, len(mu)) + 1j * np.random.normal(0, pump_noise_amp, len(mu))
            fE_in = fE_in_o + noise
            add = fE_in * h

        DV += Delta_v_add
        lin_part = -(kappa_all / kappa_avg) - 1j * DV - 1j * Dint_norm
        exp_prop = np.exp(lin_part * h)
        spectrum = exp_prop * (spectrum + add)
        signal = ifft(fftshift(spectrum))
        signal = np.exp(1j * np.abs(signal)**2 * dt) * signal
        spectrum = fftshift(fft(signal))
        spectrum = exp_prop * (spectrum + add)

        if j % plot_step == 0:
            # Track intracavity power evolution
            power = np.sum(np.abs(signal)**2) / nF
            intracavity_power_live.append((DV, power, j))

            # Plot and save spectrum frame
            current_spectrum = fftshift(fft(signal)) / nF
            plt.figure()
            
            frequencies = f0 + mu * FSR  # [Hz]
            wavelengths_nm = c / frequencies * 1e9  # λ = c / f → [nm]
            
            detuning_Hz = DV * kappa_avg / (2 * np.pi)  # Convert normalized Δ to Hz
            power_W = power * P_0  # Convert normalized intracavity power to W
            
            #plt.title(f"Spectrum at step {j}\nDetuning = {detuning_Hz*1e-9:.2f} GHz, Intracavity power = {power_W*1e6:.1f} µW")
            plt.title(f"Spectrum at step {j}\nDetuning = {DV:.2f}, Intracavity power = {power:.2f}")

            #plt.plot(mu, 10 * np.log10(np.abs(current_spectrum)**2))
            #plt.xlabel("Mode number (mu)")
            
            plt.plot(wavelengths_nm, 10 * np.log10(np.abs(current_spectrum)**2))
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Power (dB)")
            plt.ylim([-200, 20])
            plt.grid(True)
            plt.tight_layout()
            filename = f"spectrum_step_{j:06d}.png"
            plt.savefig(filename)
            plt.show()
            
            # Plot and save time-domain pulse profile
            plt.figure()
            plt.plot(tau, np.abs(np.fft.ifftshift(signal))**2)
            plt.title(f"Pulse Profile at step {j}\nDetuning = {DV:.2f}, Power = {power:.2f}")
            plt.xlabel("Fast time τ (fs)")
            plt.ylabel("Intensity")
            plt.xlim([-250, 250])  # 
            plt.ylim([0, 10])  # Adjust if needed
            plt.grid(True)
            plt.tight_layout()
            filename = f"pulse_profile_step_{j:06d}.png"
            plt.savefig(filename)
            plt.close()

    signal = ifft(fftshift(spectrum))

    file_name = f"Ring_Y_P={str(P_in).replace('.', '_')}"
    np.savez_compressed(file_name,
                        signal=signal,
                        spectrum=spectrum,
                        Save_signal=SaveSignal,
                        Save_detuning=SaveDet,
                        Disp=Dint,
                        Q=Q,
                        D2=D2,
                        P=P_in)

    slice_index = np.argmin(np.abs(SaveDet - P_in))
    soliton_spectrum = fftshift(fft(SaveSignal[slice_index, :])) / nF
    soliton_intensity = np.abs(SaveSignal[slice_index, :])**2

    # === Intracavity power vs. detuning plot with simulation step axis ===
    intracavity_power = np.sum(np.abs(SaveSignal)**2, axis=1) / nF
    steps = np.arange(1, N + 1, save_step)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twiny()


    ax1.plot(SaveDet, intracavity_power)
    ax1.set_xticks(SaveDet[::max(1, len(SaveDet)//20)])
    ax1.set_xticklabels([f"{x:.1f}" for x in SaveDet[::max(1, len(SaveDet)//20)]])
    ax1.set_xlabel("Detuning (Δ)")
    ax1.set_ylabel("Power")
    ax1.set_title("Intracavity Power vs Detuning")
    ax1.grid(True)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(SaveDet[::max(1, len(SaveDet)//10)])
    ax2.set_xticklabels([f"{steps[i]/1e6:.2f}" for i in range(0, len(steps), max(1, len(steps)//10))])
    ax2.set_xlabel("Simulation Step")

    plt.tight_layout()
    plt.savefig("intracavity_power_vs_detuning.png")
    plt.show()

# === Create animation from saved spectrum plots ===
spectrum_images = []

disp_suffix = os.path.splitext(os.path.basename(d2_file_path))[0]

for fname in sorted(glob.glob("spectrum_step_*.png")):
    spectrum_images.append(imageio.imread(fname))
imageio.mimsave(f"spectrum_evolution_{disp_suffix}.gif", spectrum_images, duration=0.8)

# === Create animation from intracavity power evolution ===
# Generate one frame per tracked intracavity power plot
power_images = []
for i in range(len(intracavity_power_live)):
    dvals = [p[0] for p in intracavity_power_live[:i+1]]
    pvals = [p[1] for p in intracavity_power_live[:i+1]]
    step_vals = [p[2] for p in intracavity_power_live[:i+1]]  # grab step indices
    det = intracavity_power_live[i][0]
    step_val = intracavity_power_live[i][2]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twiny()

    ax1.plot(dvals, pvals, color='tab:blue')
    ax1.set_xlim(SaveDet[0], SaveDet[-1])
    ax1.set_ylim(0.0, 1.6)
    ax1.set_xlabel("Detuning (Δ)")
    ax1.set_ylabel("Power")
    ax1.set_title(f"Intracavity Power vs Detuning and Simulation Step\nStep = {step_val / 1e6:.2f}×10⁶, Detuning = {det:.2f}")
    ax1.grid(True)

    # Tick setup remains unchanged

    # Fixed ticks
    fixed_xticks = np.linspace(SaveDet[0], SaveDet[-1], 10)
    fixed_yticks = np.linspace(0.0, 1.6, 9)
    fixed_xticklabels = [f"{x:.1f}" for x in fixed_xticks]
    fixed_xticklabels_top = [f"{steps[np.argmin(np.abs(SaveDet - x))] / 1e6:.2f}" for x in fixed_xticks]

    ax1.set_xticks(fixed_xticks)
    ax1.set_xticklabels(fixed_xticklabels)
    ax1.set_yticks(fixed_yticks)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(fixed_xticks)
    ax2.set_xticklabels(fixed_xticklabels_top)
    ax2.set_xlabel("Simulation Step ($\\times10^6$)")

    plt.tight_layout()
    frame_name = f"power_live_frame_{i:04d}.png"
    plt.savefig(frame_name)
    plt.close()
    power_images.append(imageio.imread(frame_name))

imageio.mimsave(f"intracavity_power_evolution_{disp_suffix}.gif", power_images, duration=0.5)

# === Create animation from saved pulse profile plots ===
pulse_images = []
for fname in sorted(glob.glob("pulse_profile_step_*.png")):
    pulse_images.append(imageio.imread(fname))

imageio.mimsave(f"pulse_profile_evolution_{disp_suffix}.gif", pulse_images, duration=0.8)

# === Intracavity power vs. detuning plot with simulation step axis ===
intracavity_power = np.sum(np.abs(SaveSignal)**2, axis=1) / nF
steps = np.arange(1, N + 1, save_step)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twiny()

# Choose consistent tick indices
tick_idx = np.arange(0, len(SaveDet), max(1, len(SaveDet) // 15))

# Bottom axis: detuning
ax1.plot(SaveDet, intracavity_power)
ax1.set_xticks(SaveDet[tick_idx])
ax1.set_xticklabels([f"{SaveDet[i]:.1f}" for i in tick_idx])
ax1.set_xlabel("Detuning (Δ)")
ax1.set_ylabel("Power")
ax1.set_title("Intracavity Power vs Detuning and Simulation Step")
ax1.grid(True)

# Top axis: simulation steps
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(SaveDet[tick_idx])
ax2.set_xticklabels([f"{steps[i] / 1e6:.2f}" for i in tick_idx])
ax2.set_xlabel("Simulation Step ($\\times10^6$)")

plt.tight_layout()
plt.savefig("intracavity_power_vs_detuning.png")
plt.show()

# === Cleanup: Remove temporary PNG files used to generate GIF animations ===

time.sleep(30)
gc.collect()

# === Cleanup: Remove temporary PNG files used to generate GIF animations ===
for fname in glob.glob("spectrum_step_*.png"):
    os.remove(fname)

for fname in glob.glob("power_live_frame_*.png"):
    os.remove(fname)
    
for fname in glob.glob("pulse_profile_step_*.png"):
    os.remove(fname)





