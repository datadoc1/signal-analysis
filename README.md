# Residency Signal Analysis Project

This repository contains all scripts and data used for our research project exploring the impact of residency application signals on interview outcomes across various medical specialties. Our goal is to expand on preliminary analyses by studying how signals affect interview rates, with a focus on comparing in-state versus out-of-state applicants, and various other factors.

## Project Overview

Residency signals were introduced to help applicants express special interest in certain programs during the application process. Initially rolled out in the 2022-2023 cycle for a few specialties like Orthopedics, signals have now been expanded across many fields. This project uses statistical analysis and Monte Carlo simulations to explore several aspects of the signal system, including:

- **How signals impact interview rates.**
- **Comparing success rates of applicants from different states.**
- **Understanding how optimal signal placement can improve match outcomes.**

Each specialty has its own unique dynamics, and this project aims to generate insights across various medical fields.

## Current Focus Areas

1. **Specialty-Specific Analysis:**
   - We're starting with Orthopedics, Dermatology, and expanding to other competitive specialties like Internal Medicine, Pediatrics, and Emergency Medicine.

2. **Monte Carlo Simulations:**
   - Simulation of interview outcomes based on different signaling strategies, with a specific focus on comparing high-signal programs to random signaling.

3. **Statistical Breakdown:**
   - Analysis of residency interview odds based on variables such as signal placement, applicant residency (in-state vs. out-of-state), medical degree (MD, DO), and IMG status.

## Repository Structure

- **specialties_data/**
  - This folder contains the raw data files for different medical specialties.
  
- **scripts/**
  - Contains the Python scripts for data processing, statistical analysis, and Monte Carlo simulations. Current scripts focus on Dermatology but will be expanded to other fields soon.

## How to Contribute

Each manuscript team will be working on a specific aspect of the data for their respective specialty. The current active manuscripts are:

### Manuscripts in Progress

1. **Optimal Signaling in Orthopedics**
   - **Authors:** [Name1], [Name2]
   - **Synopsis:** Analyzing how the distribution of signals affects interview rates for Orthopedic residency programs, focusing on the potential benefits of strategic signaling.
  
2. **Dermatology Residency Signal Analysis**
   - **Authors:** [Name1], [Name2]
   - **Synopsis:** A statistical evaluation of how Dermatology residency programs respond to signals, with a focus on how in-state vs. out-of-state applicants fare in securing interviews.

3. **Signal Impact in Emergency Medicine**
   - **Authors:** [Name1]
   - **Synopsis:** Emergency Medicine's response to signals, examining how program-specific factors affect interview chances for different applicant types.

4. **Geographic Bias in Residency Interviews**
   - **Authors:** [Name1], [Name2]
   - **Synopsis:** How state residency influences interview opportunities across multiple specialties, analyzing which states confer the greatest advantage to in-state applicants.

### Adding New Manuscripts

If you are part of a team working on a new manuscript, please follow this format to add your project to the README:
- **Title:** 
- **Authors:** 
- **Synopsis:** Brief description of the manuscript and the area of focus.

## How to Access and Use the Data

Data files for each specialty are stored in the `specialties_data/` folder. The script folder includes a Python file that loads and processes the CSV files and runs Monte Carlo simulations to evaluate signaling outcomes.

### Example Script

```python
import pandas as pd
import random

# Load and clean data
df = pd.read_csv('specialties_data/dermatology.csv')
# Process numeric fields and run simulation
...
```

The above script is a basic example for Dermatology, showing how to load the data and perform basic simulations. Results will include interview rate distributions and confidence intervals for both gold and silver signals.

## Tools and Resources

We are using **Zotero** for citation management and **SciSpace** for exploring relevant academic papers. Please ensure that all manuscripts are properly cited and that literature reviews are thorough and up-to-date.

### Getting Started with Zotero

If you're new to Zotero, instructions on how to install and use it are available [here](https://www.zotero.org/support/quick_start_guide). For integrating the Zotero Chrome extension and tools like SciSpace for literature management, refer to our [workflow guide](./workflow_guide.md).

## Contact

If you have any questions or need additional resources, feel free to reach out directly. All coding tasks are handled separately, and I will continue updating scripts as needed for each specialty.