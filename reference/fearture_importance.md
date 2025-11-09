# Feature Selection for Final Analysis

| Feature | Important | Reasons |
|---|---|---|
| **Encounter ID** | No | Used only for initial data extraction; not a predictor of outcome. |
| **Patient number** | No | Used to ensure only one encounter per patient was analyzed; not a predictor itself. |
| **Race** | Yes | Kept as a demographic covariate in the logistic regression model. |
| **Gender** | No | Initially considered but found to be statistically insignificant (P = 0.36) and was removed from the core model. |
| **Age** | Yes | Kept as a demographic covariate. It was grouped into three categories ([0,30), [30,60), [60,100)) based on preliminary analysis. |
| **Weight** | No | Removed due to an extremely high percentage of missing values (97%). |
| **Admission type** | Yes | Kept as a covariate representing the type of admission. |
| **Discharge disposition** | Yes | Kept as a covariate, with a strong influence on readmission probability (e.g., discharged to home vs. other). |
| **Admission source** | Yes | Kept as a covariate in the model. |
| **Time in hospital** | Yes | Kept as a numeric covariate representing the length of stay. |
| **Payer code** | No | Removed due to a high percentage of missing values (52%) and was not considered relevant to the outcome. |
| **Medical specialty** | Yes | Kept as a covariate. Missing values were handled by adding a "missing" category. |
| **Number of lab procedures** | No | Removed during the preliminary analysis to focus on features more directly related to diabetes management and readmission. |
| **Number of procedures** | No | Removed during the preliminary analysis to focus on features more directly related to diabetes management and readmission. |
| **Number of medications** | No | Removed during the preliminary analysis to focus on features more directly related to diabetes management and readmission. |
| **Number of outpatient visits** | No | Removed during the preliminary analysis to focus on features more directly related to diabetes management and readmission. |
| **Number of emergency visits** | No | Removed during the preliminary analysis to focus on features more directly related to diabetes management and readmission. |
| **Number of inpatient visits** | No | Removed during the preliminary analysis to focus on features more directly related to diabetes management and readmission. |
| **Diagnosis 1 (Primary Diagnosis)** | Yes | Kept as a key covariate. The study found a significant interaction between primary diagnosis and HbA1c measurement. |
| **Diagnosis 2** | No | Removed to simplify the model; the primary diagnosis was deemed sufficient to represent the patient's main condition. |
| **Diagnosis 3** | No | Removed to simplify the model; the primary diagnosis was deemed sufficient to represent the patient's main condition. |
| **Number of diagnoses** | No | Removed to simplify the model; the primary diagnosis was deemed sufficient to represent the patient's main condition. |
| **Glucose serum test result** | No | The study's primary focus was on HbA1c as a marker of care quality, not on individual serum glucose results. |
| **A1c test result** | Yes | This is the main feature of interest. It was categorized into four groups: not measured, normal, high with medication change, and high with no change. |
| **Change of medications** | Yes | This feature was used to define the categories for the `A1c test result` variable (e.g., "high, changed"). |
| **Diabetes medications** | Yes | Kept as it indicates the patient's diabetic status and treatment during the encounter. |
| **24 features for medications** | No | These were aggregated into the `Change of medications` and `Diabetes medications` features for a simpler, more interpretable model. |
| **Readmitted** | Yes | This is the outcome variable (the dependent variable) that the model aimed to predict. |
