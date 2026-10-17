# Source data

`Data_Parastar_corrected.csv` contains the corrected public chicken-meat NIR
dataset: 4,053 spectra from 153 fillets, with 125 wavelength variables.

Source: Parastar et al. (2020), *Integration of handheld NIR and machine learning
to “Measure & Monitor” chicken meat authenticity*, Food Control, 112, 107149.
Dataset: https://data.mendeley.com/datasets/cp2hdhkys3/6

Columns: Sample_number, Production_system, Scan_type, Freshness, followed by
wavelength variables. FR = fresh, TH = thawed. Production_system is not a model
feature. All retained released spectra are used; missing measurements are not
imputed for model training or evaluation.
