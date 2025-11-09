# Task 1a: Data Preparation - Predictor Selection and Justification

## Objective

The primary goal of this task is to critically analyze the diabetic_data.csv dataset to select a subset of relevant predictors for building a neural network model. The selection process must be justified based on clinical relevance, insights from the provided research paper ("Impact of HbA1c Measurement on Hospital Readmission Rates"), and data quality considerations.

This document provides a step-by-step guide to perform this analysis and prepare the written content for your assessment report.

## Step 1: Initial Data Exploration and Categorization

Before any selection, you must understand the full scope of the data available.

### Instructions for step 1

**Load the Data**: Load the diabetic_data.csv file into a pandas DataFrame.

**Inspect the Structure**: Display the first few rows, the data types (.info()), and a summary of all columns.

**Categorize Variables**: Create a mental or written list of all variables and group them into logical categories. This systematic approach ensures you consider all features.

### Suggested Variable Categories

- **Patient Demographics**: race, gender, age
- **Hospital Encounter Details**: admission_type_id, discharge_disposition_id, admission_source_id, time_in_hospital
- **Clinical Severity & Utilization**: num_lab_procedures, num_procedures, num_medications, number_diagnoses
- **Patient History (Pre-Admission)**: number_outpatient, number_emergency, number_inpatient
- **Diabetes-Specific Metrics**: max_glu_serum, A1Cresult, change, diabetesMed
- **Diabetes Medications**: The 24 columns for specific drugs (e.g., metformin, insulin)
- **Diagnosis Codes**: diag_1, diag_2, diag_3
- **Administrative & Provider**: payer_code, medical_specialty
- **Identifiers**: encounter_id, patient_nbr

## Step 2: Apply an Exclusion Filter

Systematically identify and document variables that are unsuitable for a predictive model. This is a critical data cleaning step.

### Instructions for step 2

#### **Identify Identifiers:**

**Variables**: encounter_id, patient_nbr

**Action**: Plan to exclude these.

**Justification for Report**: These are unique identifiers with no predictive power. Including them would cause the model to memorize specific patients or visits, leading to severe overfitting and poor generalization.

#### **Identify Variables with Excessive Missing Data:**

**Action**: Calculate the percentage of missing values for each column. Note that missing values in this dataset are often represented by a '?'.

**Variables to Target**: weight, payer_code.

**Justification for Report**: The research paper explicitly states that weight is ~97% missing and payer_code is ~52% missing. Imputing such a large portion of the data would introduce significant noise and bias. The original authors removed these features for this reason, and we will follow their evidence-based approach.

#### **Identify Redundant or High-Cardinality Variables (for initial model):**

**Variables**: diag_2, diag_3.

**Action**: Plan to exclude these for the initial model.

**Justification for Report**: The primary diagnosis (diag_1) is the main driver of the hospital stay. Secondary diagnoses add complexity (high cardinality) and can be incorporated later. For a baseline model, focusing on diag_1 provides a more manageable starting point.

## Step 3: Select and Justify Core Predictors

From the remaining variables, select a robust set of predictors. Justify each choice based on clinical logic and the findings of the research paper.

### Instructions for step 3

#### **Select Tier 1 (Essential) Predictors**: These are non-negotiable based on the project's context

##### **A1Cresult**

**Justification**: This is the central variable of the research paper. The entire study is about the "Impact of HbA1c Measurement." This variable represents long-term glucose control and the level of clinical attention to diabetes. It is the single most important feature to include.

##### **time_in_hospital**

**Justification**: A direct proxy for the severity and complexity of the patient's condition during the admission. Longer stays are strongly correlated with higher readmission risk.

##### **num_medications**

**Justification**: Indicates the complexity of the patient's treatment regimen and the potential burden of comorbidities, which are key factors in post-discharge stability.

##### **diag_1 (Primary Diagnosis)**

**Justification**: The reason for admission is a fundamental predictor. The paper explicitly states that the relationship between HbA1c and readmission depends on the primary diagnosis. This variable must be included, likely after grouping the ICD-9 codes into broader clinical categories (e.g., "Circulatory", "Respiratory", "Diabetes") as done in the paper.

#### **Select Tier 2 (Important Supporting) Predictors**: These provide crucial context

##### **age**

**Justification**: A classic demographic predictor in healthcare. Older patients generally have higher readmission risk due to physiological factors and potential for less social support.

##### **discharge_disposition_id**

**Justification**: The destination after discharge (e.g., home, skilled nursing facility) is a strong indicator of the patient's functional status and support system, directly impacting readmission likelihood.

##### **number_inpatient (prior year)**

**Justification**: A patient's recent history of hospitalizations is one of the most powerful predictors of future hospitalizations, indicating a pattern of health instability.

#### **Select Tier 3 (Contextual) Predictors**: These can add nuance but are less critical

##### **change (of medications)**

**Justification**: This variable indicates whether the clinical team actively intervened. The paper noted a correlation between HbA1c testing and medication changes. This can be a proxy for proactive, high-quality care.

##### **medical_specialty**

**Justification**: The admitting physician's specialty can influence treatment protocols. The paper found it to be significant. However, it requires careful handling due to many categories and missing values (e.g., group rare specialties, create a "Missing" category).

## Step 4: Prepare the Written Justification for Your Report

Structure your findings into a clear, professional section for your final PDF report.

### Instructions for step 4

#### **Create a Section Title**: 1a) Selection of Relevant Predictors

#### **Write an Introduction**: Briefly state the objective of the predictor selection process

#### **Document Exclusions**

Create a subsection titled **Variables Excluded from Analysis**.
Use a bulleted list or a small table.
For each excluded variable (weight, payer_code, encounter_id, patient_nbr), provide a concise but clear justification based on the reasoning in Step 2.

#### **Document Inclusions**

Create a subsection titled **Selected Predictors and Their Rationale**.
Organize your selected predictors into the tiers described in Step 3 (Essential, Important, Contextual).
For each predictor (or logical group of predictors), write a 2-3 sentence paragraph explaining why it was selected. Use phrases like:

- "This variable is a key measure of..."
- "Based on the findings of Strack et al. (2014), this variable is critical because..."
- "Clinically, this is understood to be a proxy for..."
- "Including this variable helps control for..."

#### **Concluding Summary**: End with a brief sentence summarizing that the selected features provide a comprehensive and clinically-grounded foundation for the neural network model

## Final Deliverable for 1a

Your final output for this task is the written text for your report, as described in Step 4. This text should be clear, well-justified, and demonstrate a deep understanding of both the data and the clinical problem. The coding work you do in Steps 1-3 is to support the analysis needed to write this justification.
