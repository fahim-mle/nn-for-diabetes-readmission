# Detailed Plan for Task 1a: Selecting and Justifying Predictors

The goal of this task is to demonstrate that you have critically examined the available data and made informed decisions about which features are most likely to be predictive of hospital readmission. You need to go beyond simply listing variables and provide a clear rationale for your choices, linking them back to the clinical context provided in the BioMed research paper.

## Step 1: Understand the Core Prediction Objective

Before selecting any predictors, you must clearly define what you are trying to predict and why.

**Action**: Re-read the assignment and the research paper's abstract and introduction.

### Key Questions to Answer

- What is the specific outcome we are predicting? (e.g., readmission within 30 days, yes/no)
- What does a "readmission" signify in a clinical context? (It often signals a failed treatment plan, poor disease management, complications, or inadequate post-discharge support)
- What was the central hypothesis of the original research paper? (That measuring HbA1c, a marker of attention to diabetes, is associated with lower readmission rates)

**Output for Your Report**: A concise paragraph in your report stating the prediction goal and its clinical significance. This sets the stage for your predictor selection.

## Step 2: Categorize All Available Variables

You cannot decide what to keep without first understanding what you have. Group the variables from the data description into logical categories. This systematic approach ensures you don't miss important features.

**Action**: Create a table or a list in your notes, categorizing the variables. Use the data description from the assignment (Table 1) and the paper.

### Suggested Categories

- **Patient Demographics**: race, gender, age
- **Hospital Encounter Details**: encounter_id, patient_nbr, admission_type_id, discharge_disposition_id, admission_source_id, time_in_hospital
- **Clinical Severity & Utilization**: num_lab_procedures, num_procedures, num_medications, number_diagnoses
- **Patient History**: number_outpatient, number_emergency, number_inpatient (in the year prior)
- **Diabetes-Specific Care & Metrics**: max_glu_serum, A1Cresult, change, diabetesMed, and the 24 medication features
- **Administrative & Provider Information**: payer_code, medical_specialty
- **Primary Diagnosis**: diag_1, diag_2, diag_3
- **Identifiers**: encounter_id, patient_nbr

## Step 3: Apply an Exclusion Filter

Now, systematically eliminate variables that are clearly unsuitable for a predictive model. This simplifies your selection process.

**Action**: Go through your categorized list and apply the following rules. Document your reasons for exclusion.

### Exclusion Criteria and Rationale

#### **Identifiers**: encounter_id, patient_nbr

**Reason**: These are unique keys for records and people. They contain no predictive information about the outcome and would only cause the model to memorize specific patients or visits, leading to massive overfitting.

#### **Variables with Excessive Missing Data**: weight, payer_code

**Reason**: The paper explicitly states that weight is 97% missing and payer_code is 52% missing. Imputing such a large percentage of data would introduce significant noise and bias. The authors of the original paper themselves removed weight and payer_code for this reason. You should follow their evidence-based decision.

#### **Redundant Variables**: diag_2 and diag_3 (initially)

**Reason**: The primary diagnosis (diag_1) is the most important driver of the hospital stay and treatment plan. While secondary diagnoses are useful, they can dramatically increase the complexity of your model (high cardinality). For a baseline model, it's reasonable to start with only the primary diagnosis and add the others later if needed.

## Step 4: Select and Justify Your Core Predictors

This is the most critical step. From the remaining variables, select a subset that is both clinically relevant and manageable for a neural network. Group them into tiers of importance.

**Action**: Select your final list of predictors and write a justification for each one, or for logical groups of predictors.

### Tier 1: The Essential Predictors (Must-Haves)

These are variables that are either central to the research question or are strong, direct indicators of patient condition.

#### **A1Cresult (HbA1c test result)**

**Justification**: This is the most important predictor based on the entire premise of the research paper. The paper's title is "Impact of HbA1c Measurement on Hospital Readmission Rates." Your model must include this variable to test the paper's core hypothesis. It represents the long-term blood sugar control of the patient and the level of clinical attention paid to their diabetes during the hospital stay.

#### **time_in_hospital**

**Justification**: A direct measure of the severity and complexity of the patient's condition during the admission. Longer stays are often associated with more severe illness, more complications, and a higher risk of readmission.

#### **num_medications**

**Justification**: Indicates the complexity of the patient's treatment regimen. A higher number of medications can suggest more comorbidities (other health problems) and a higher risk of drug interactions or non-adherence after discharge.

#### **diag_1 (Primary Diagnosis)**

**Justification**: The primary reason for the hospitalization is a fundamental predictor of outcomes. As the paper found, the relationship between HbA1c and readmission depends on the primary diagnosis. For example, a patient admitted for a circulatory issue has a different risk profile than one admitted for an infection. This variable must be included, likely after grouping the ICD-9 codes into broader categories (e.g., Circulatory, Respiratory, Diabetes) as done in the paper.

### Tier 2: Important Supporting Predictors

These variables provide crucial context about the patient's baseline health and recent healthcare utilization.

#### **age**

**Justification**: Age is a classic predictor in healthcare. Older patients generally have more fragile health, slower recovery, and a higher likelihood of readmission.

#### **discharge_disposition_id**

**Justification**: Where a patient goes after discharge is highly predictive. A patient discharged to a skilled nursing facility is at a different risk level than one discharged home with self-care.

#### **number_inpatient (prior year)**

**Justification**: A patient's recent history of hospitalizations is one of the strongest predictors of future hospitalizations. It indicates a pattern of health instability.

### Tier 3: Potentially Useful Contextual Predictors

These variables can add nuance but are less critical than the tiers above. You could include them in a more advanced model.

#### **change (of medications)**

**Justification**: This directly measures whether the clinical team intervened based on the patient's condition (e.g., their HbA1c or glucose levels). The paper notes that providers who ordered HbA1c were more likely to change medications. This could be a proxy for proactive care.

#### **num_lab_procedures**

**Justification**: Similar to time_in_hospital, this can be a proxy for the intensity of investigation and the severity of the patient's condition.

#### **medical_specialty**

**Justification**: The admitting physician's specialty can influence treatment patterns and patient management. The paper found it to be a significant variable. However, it has many categories and a high percentage of missing values, so it requires careful handling (e.g., creating a "Missing" category and grouping less common specialties).
