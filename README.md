# Residency Signal Analysis Project

This repository contains all scripts and data used for our research project exploring the impact of residency application signals on interview outcomes across various medical specialties. Our goal is to expand on preliminary analyses by studying how signals affect interview rates, with a focus on comparing in-state versus out-of-state applicants, and various other factors.

## Project Overview

Residency signals were introduced to help applicants express special interest in certain programs during the application process. Initially rolled out in the 2022-2023 cycle for a few specialties like Orthopedics, signals have now been expanded across many fields. This project uses statistical analysis and Monte Carlo simulations to explore several aspects of the signal system, including:

- **How signals impact interview rates.**
- **Comparing success rates of applicants from different states.**
- **Understanding how optimal signal placement can improve match outcomes.**

Each specialty has its own unique dynamics, and this project aims to generate insights across various medical fields.

## Current Focus Areas
1. **Statistical Breakdown:**
* Analysis of residency interview odds based on variables such as signal placement, applicant residency (in-state vs. out-of-state), medical degree (MD, DO), and IMG status.
  
2. **Monte Carlo Simulations:**
* Simulation of interview outcomes based on different signaling strategies, with a specific focus on comparing optimal signaling to random signaling.
  
3. **Geographical Biases**
* Analysis of interview outcomes based on different states of residence. Which states give in-state students the largest edge?


## How to Access and Use the Data

- **specialties_data/**
  - This folder contains the raw data files for different medical specialties.
  
- **reports/**
  - Contains data reports from completed analysis

### Interpreting the Reports
- **descriptive_stats.csv**
  - 'Count' represents the number of programs that reported data on that column of data
  - 'Mean' represents the odds of getting an interview at the average program
  - 'Min' and 'Max' represent the interview odds of the programs with the lowest and highest interview odds
  - '25%', '50%', '75%' represent percentiles. The median program has an interview rate represented by '50%'
  - All of these rows are stratified by the columns

- **geographic_bias.csv**
  - *For the purposes of this data, being "in-state" means going to a medical school in that state, someone born in TX who goes to Yale is a CT resident, not a TX resident*
  - 'In-State School Mean' represents the mean interview odds for in-state students for programs in each state. 0.42 for AL means that an AL resident has 42% odds of landing an interview at the average AL medical school.
  - 'In-State School Sum' represents the total interviews an in-state student should expect to get if they apply to every school in their state. 0.83 for AL means that an AL resident should expect 0.83 interviews if they apply to every program in AL.
  - 'Out-Of-State School Mean' represents the mean interview odds for out-of-state students for programs in each state. 0.05 for AL means that a non-AL resident has 5% odds of landing an interview at the average AL medical school
  - 'Out-Of-State School' represents the total interviews an out-of-state student should expect to get if they apply to every school in that state. 0.1 for AL means that a non-AL resident should expect 0.1 interviews if they apply to every program in AL.
  - 'Mean Difference' represents the difference between In-State and Out-Of-State mean. Higher values indicate higher preference for in-state students.
  - 'Sum Difference' represents the difference between In-State and Out-Of-State sum. A value of 0.73 for AL indicates that if we took two otherwise identical applicants, one from a medical school in AL and one from a non-AL medical school and had them apply to every medical school in AL, we should expect the student at the AL medical school to get 0.73 more interviews.
 
- **optimal_signals.txt**
  - This file represents the optimal signals including programs and interview likelihood along with total interviews expected if a completely average applicant signalled at these schools.

- **random_simulation.txt**
  - This file represents the number of interviews one could expect to recieve if they randomly chose signals. A 95% CI is included.

## Manuscripts in Progress

[Quantifying the Impact of ERAS Signals and State Residency on Radiology Interview Rates](https://docs.google.com/document/d/1dtocSvL4ES1k8dKN3zMrgUB0OsSqpPiwgt2l1gGgpIs/edit?tab=t.0)

## Tools and Resources

We are using [Zotero](https://www.zotero.org/groups/5692550/eras-signals/library) for citation management and **SciSpace/Elicit** for exploring relevant academic papers. Please ensure that all manuscripts are properly cited and that literature reviews are thorough and up-to-date.

## Contact

If you have any questions or need additional resources, feel free to reach out directly. All coding tasks are handled separately, and I will continue updating scripts as needed for each specialty.
