# [Real-Time Power Side-Channel Defense Against UAV Vision Backdoors: Physics-Aware Digital Twin]

This repository contains the core implementation of the PADT framework and SSP defense architecture proposed in our paper.

---

### Repository Structure
* **`Core_Logic/`**: Implementation of the integer-domain ZOH and PADT synchronization.
* **`Hardware_Config/`**: STM32 firmware source and ADC sampling logic.
* **`Inference/`**: Scripts for Jetson-based vision tasks.
* **`Sample_Data/`**: A representative subset of power consumption traces for functional verification.

---

### Reviewer Note
To comply with the **double-blind review** and maintain a lightweight repository, this version provides:
1.  **Core Algorithms:** All critical logic for the physical-layer auditing paradigm.
2.  **Representative Data:** A subset of power traces to demonstrate the detection of pixel-level triggers.

**The full dataset (multi-GB), complete codebase, and pre-trained models will be made publicly available upon formal publication.**

---

### Anonymity
This repository is hosted via **Anonymous GitHub**. All metadata and file paths have been sanitized to preserve the integrity of the double-blind review process.
