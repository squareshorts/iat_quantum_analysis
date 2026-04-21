# Data Availability

The manuscript's main analysis uses public Project Implicit demo-release trial logs stored locally in:

- `data/GenderScience_iat_2019/iat_2019/iat.txt`
- `data/GenderScience_iat_2019/iat_2019/iat2.txt`
- `data/GenderScience_iat_2019/iat_2019/iat.0003.txt`

The main pipeline reads these files via:

- `run_theta_grid_full_plus_baselines.py`
- `run_theta_grid_full.py`

Additional local datasets in `data/race/` and `data/age_netherlands/` are exploratory cross-domain materials and are not required for the manuscript's core Gender--Science result.

An independent 2024 life-satisfaction IAT archive used for the manuscript's external-leverage analyses is stored locally in:

- `data/life_satisfaction_iat_2024/raw/187results.zip`

The corresponding organization/cleaning pipeline is:

- `run_external_leverage_analysis.py`

That script reconstructs the 180 valid participants described in the project notes and exports anonymized processed files to:

- `outputs/external_life_satisfaction_participants_clean.csv`
- `outputs/external_life_satisfaction_trials_clean.csv`
- `outputs/external_life_satisfaction_excluded_cases.csv`

Supporting local metadata:

- `data/GenderScience_iat_2019/iat_RawData_Codebook.xlsx`

Before external release or journal submission, add the exact public dataset citation and access URL required by the target venue.
