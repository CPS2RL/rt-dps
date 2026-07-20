
# Understanding the Pitfalls of a Differentially Private Fixed-Priority Real-Time Scheduler

This repository contains the source code and experiment scripts used to reproduce the results of the EMSOFT 2026 paper: **Understanding the Pitfalls of a Differentially Private Fixed-Priority Real-Time Scheduler** [PDF](https://monowarhasan.info/papers/RT-DPS_Pitfalls_EMSOFT26.pdf).

**Authors:** Mohammad Fakhruddin Babar, Tamim Ahmed, and Monowar Hasan

## Table of Contents

* [Objective of the Paper](#objective-of-the-paper)
* [Attack Patterns](#attack-patterns)
* [Repository Structure](#repository-structure)
* [Environment Setup](#environment-setup)

  * [Linux](#linux)
  * [Windows](#windows)
* [Running the Experiments](#running-the-experiments)

  * [Anterior, Posterior, and Pincer Attacks](#anterior-posterior-and-pincer-attacks)
  * [Control-Stability Analysis](#control-stability-analysis)
  * [Normalized Attack Ratio](#normalized-attack-ratio)
  * [Sensitivity-Parameter Sweep](#sensitivity-parameter-sweep)
  * [Example Schedule Plot](#example-schedule-plot)
  * [Jitter-Margin Analysis](#jitter-margin-analysis)
* [Generated Outputs](#generated-outputs)
* [Reproducibility Notes](#reproducibility-notes)
* [Citation](#citation)

## Objective of the Paper

Differentially private real-time schedulers randomize task inter-arrival times to reduce the predictability of execution patterns. Although this randomization can improve timing privacy, it may also introduce unexpected security and control-system consequences.

This work investigates the behavior of a differentially private fixed-priority scheduler, referred to as **DPS**, and compares it with a conventional deterministic Rate-Monotonic scheduler, referred to as **Vanilla RM** or **TS**.

The experiments evaluate:

1. Whether DPS creates or removes schedule-based attack opportunities.
2. How frequently anterior, posterior, and pincer attack patterns occur.
3. How attack intensity changes after normalization by the number of attacker and victim jobs.
4. How the inter-arrival-time sensitivity parameter affects attack opportunities.


## Attack Patterns

This repository evaluates three schedule-based attack patterns:

* **Anterior attack:** The attacker executes immediately before the victim.
* **Posterior attack:** The attacker executes immediately after the victim.
* **Pincer attack:** The same attacker job executes immediately before and immediately after the victim.

The experiments compare the occurrence of these attack patterns under Vanilla RM and DPS.

## Repository Structure

```text
rt-dps/
├── README.md
├── ap_ant_post_pin.py
├── ap_butterfly.py
├── normalized.py
├── delta_eta_sweep_attack_plot.py
├── example_plot/
│   └── example_timing_plot_generation.ipynb
└── jitter_margin/
    └── jitter margin curve.ipynb
```

### Main Files

* `ap_ant_post_pin.py`
  Evaluates anterior, posterior, and pincer attack opportunities under Vanilla RM and DPS.

* `ap_butterfly.py`
  Evaluates response-time variation and control-stability violations under Vanilla RM and DPS.

* `normalized.py`
  Calculates normalized attack ratios using the number of attacker and victim jobs.

* `delta_eta_sweep_attack_plot.py`
  Evaluates how attack opportunities change as the inter-arrival-time sensitivity parameter, Δη, is varied.

* `example_plot/example_timing_plot_generation.ipynb`
  Generates the example schedule visualization used in the paper.

* `jitter_margin/jitter margin curve.ipynb`
  Generates the jitter-margin curve for the DC motor control-system example.

## Environment Setup

The experiments are implemented in Python. We recommend running them in a virtual environment.

First, clone the repository:

```bash
git clone https://github.com/CPS2RL/rt-dps.git
cd rt-dps
```

### Linux

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required Python packages:

```bash
pip install pandas openpyxl matplotlib numpy scipy jupyter control
```

### Windows

Create and activate a Python virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install pandas openpyxl matplotlib numpy scipy jupyter control
```

## Running the Experiments

> **Note:** The scripts execute hundreds of task-set simulations. The execution time depends on the number of tasks, utilization levels, simulation horizon, number of trials, and hardware configuration.

### Anterior, Posterior, and Pincer Attacks

The `ap_ant_post_pin.py` script compares schedule-based attack opportunities under Vanilla RM and DPS.

The default experiment:

* Sweeps total utilization from 0.1 to 0.8.
* Generates 200 task sets for each utilization level.
* Simulates 200 scheduling windows.
* Randomly selects an attacker and victim task.
* Counts anterior, posterior, and pincer attack instances.
* Records the number of unique execution traces.
* Compares attack incidence under Vanilla RM and DPS.

Run the experiment using:

```bash
python ap_ant_post_pin.py
```

The number of tasks, utilization values, DPS parameters, simulation horizon, number of trials, and attacker–victim priority relationship can be changed in the main experiment block of the script.

The attacker–victim selection can be configured to evaluate:

* A higher-priority attacker and lower-priority victim.
* A lower-priority attacker and higher-priority victim.
* A randomly selected attacker–victim pair.

### Control-Stability Analysis

The `ap_butterfly.py` script evaluates whether response-time variation under Vanilla RM and DPS violates a specified control-stability constraint.

For a randomly selected target task, the script calculates:

* Best-case response time.
* Worst-case response time.
* Response-time jitter.
* Stability-constraint value.
* Whether the target task violates the stability condition.

Each task set is classified into one of four categories:

* `Vanilla_only`
* `EPS_only`
* `Both`
* `Neither`

Run the experiment using:

```bash
python ap_butterfly.py
```

The default experiment evaluates utilization levels from 0.1 to 0.8 using 200 task sets for each utilization level.

The control-stability parameters can be changed in the script:

```python
alpha = 1.20
beta = 20.83
```

The stability condition is evaluated using:

```text
R_best + alpha × (R_worst - R_best) ≤ beta
```

### Normalized Attack Ratio

The `normalized.py` script calculates normalized attack metrics.

Instead of reporting only the total number of detected attack instances, the script normalizes the attack count using the number of attacker and victim jobs:

```text
N = J_attacker + J_victim
```

The normalized attack ratio is calculated as:

```text
X = Number of attack instances / N
```

Run the experiment using:

```bash
python normalized.py
```

The script stores individual task-set results and summary statistics for every utilization level.

### Sensitivity-Parameter Sweep

The `delta_eta_sweep_attack_plot.py` script studies how the scheduler's inter-arrival-time sensitivity parameter affects schedule-based attack opportunities.

The default experiment uses:

```text
Total utilization: 0.5
Number of tasks: 10
Number of task sets: 200
Privacy parameter ε: 10
J: 16
Δη values: 0, 20, 40, ..., 200 ms
```

Run the experiment using:

```bash
python delta_eta_sweep_attack_plot.py
```

The same generated task sets and attacker–victim pairs are used across all Δη values to provide a consistent comparison.

The generated results are stored in:

```text
delta_eta_sweep_results/
```

The output files include:

```text
summary_delta_eta.csv
delta_eta_sweep_plot.pdf
delta_eta_sweep_plot.png
```

### Example Schedule Plot

The example schedule used in the paper can be generated using the notebook in the `example_plot` directory.

Run:

```bash
jupyter notebook "example_plot/example_timing_plot_generation.ipynb"
```

Execute all notebook cells to generate the example timing diagram.

### Jitter-Margin Analysis

The DC motor control-system jitter-margin curve can be generated using the notebook in the `jitter_margin` directory.

Run:

```bash
jupyter notebook "jitter_margin/jitter margin curve.ipynb"
```

Execute all notebook cells to calculate and visualize the relationship between latency and allowable response-time jitter.

## Generated Outputs

| Experiment                              | Script                                 | Output                                                |
| --------------------------------------- | -------------------------------------- | ----------------------------------------------------- |
| Anterior, posterior, and pincer attacks | `ap_ant_post_pin.py`                   | `trace_attack_counts_aRP_n5.xlsx`                     |
| Control-stability analysis              | `ap_butterfly.py`                      | `stability_violation_n15.xlsx`                        |
| Normalized attack analysis              | `normalized.py`                        | `new_exp_n15.xlsx`                                    |
| Δη sensitivity analysis                 | `delta_eta_sweep_attack_plot.py`       | CSV, PDF, and PNG files in `delta_eta_sweep_results/` |
| Example schedule                        | `example_timing_plot_generation.ipynb` | Example timing-plot figure                            |
| Jitter-margin analysis                  | `jitter margin curve.ipynb`            | Jitter-margin curve                                   |

The generated Excel workbooks contain separate sheets for individual task-set results and aggregated summaries.

## Reproducibility Notes

The experiment scripts use randomized task-set generation and randomized attacker–victim selection. Random seeds are recorded in the generated results or printed in the terminal so that individual runs can be reproduced.

Important experiment parameters are defined in each script, including:

```text
Number of tasks
Total utilization
Number of task sets
Number of simulation windows
Hyperperiod cap
Privacy parameter ε
J parameter
Sensitivity parameter Δη
Period range
Attacker and victim priority relationship
```

To reproduce a specific configuration from the paper, verify these parameters before running the corresponding experiment.

The scripts use UUniFast to generate task utilizations and Rate-Monotonic priority assignment to schedule the generated real-time tasks.

## Citation

When using this repository, please cite:

```bibtex
@inproceedings{babar2026pitfalls,
  author    = {Mohammad Fakhruddin Babar and
               Tamim Ahmed and
               Monowar Hasan},
  title     = {Understanding the Pitfalls of a Differentially Private
               Fixed-Priority Real-Time Scheduler},
  booktitle = {International Conference on Embedded Software (EMSOFT)},
  year      = {2026}
}
```
