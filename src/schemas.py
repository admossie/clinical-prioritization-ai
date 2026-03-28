COMMON_BASE_FEATURES=["race","gender","age","admission_type_id","discharge_disposition_id","admission_source_id","time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_outpatient","number_emergency","number_inpatient","number_diagnoses"]
TEMPORAL_FEATURES=["encounter_number","prior_encounters","prior_number_inpatient_sum","prior_number_inpatient_mean","prior_number_outpatient_sum","prior_number_outpatient_mean","prior_number_emergency_sum","prior_number_emergency_mean","prior_total_visits","diag_delta","med_delta","prior_positive_count","ever_prior_positive"]
ID_COLUMNS=["encounter_id","patient_nbr"]
TARGET_SOURCE_COLUMN="readmitted"
TARGET_COLUMN="target"
