# Feature Engineering Guide

## 1. Understanding the Raw Data (The 24 Features)

For each of the 24 specific diabetes medications (like metformin, insulin, glipizide, etc.), the dataset had a feature with four possible values:

- **up**: The dosage of that specific drug was increased during the hospital stay
- **down**: The dosage of that specific drug was decreased
- **steady**: The dosage of that specific drug did not change
- **no**: That specific drug was not prescribed to the patient at all

## 2. Creating the Diabetes Medications Feature

This was the simplest feature to derive.

### Logic for creating the `diabetes medications feature`

To determine if a patient was on any diabetes medication, you just need to check if at least one of the 24 medication features had a value other than **no**.

### Process for `diabetes medication feature`

1. For a given patient encounter, scan all 24 medication columns
2. If you find even a single column with a value of **up**, **down**, or **steady**, then the **Diabetes medications** feature for that encounter is **yes**
3. If all 24 columns have the value **no**, then the **Diabetes medications** feature is **no**

## 3. Creating the Change of Medications Feature

This feature was slightly more complex and looked for any change in the overall diabetes treatment plan.

### Logic for `Change of Medications Feature`

A "change" was defined as any modification to a medication, which could be a dosage change (**up** or **down**) or the introduction of a new drug.

### Process for `Change of Medications Feature`

1. For a given patient encounter, scan all 24 medication columns
2. If you find any column with the value **up** or **down**, it means a dosage was changed. The **Change of medications** feature is **change**
3. The paper also mentions "change to a drug with a different generic name." This would be identified if a patient had one drug marked as **no** (or wasn't on it before) and another drug marked as **up** or **steady**. In the context of this dataset, finding any **up** or **down** is the most direct signal of a change
4. If all prescribed medications (all columns that are not **no**) have the value **steady**, then the **Change of medications** feature is **no change**

## 4. The Most Important Aggregation: Defining the HbA1c Groups

This is where the aggregation was most critical for the study's main analysis. The researchers didn't just look at **Change of medications** in isolation. They combined it with the **A1c test result** to create the key independent variable for their model.

Recall the four groups they defined for their analysis:

1. **No HbA1c test performed**
2. **HbA1c performed and in normal range**
3. **HbA1c performed and result is greater than 8% with no change in diabetic medications**
4. **HbA1c performed, result is greater than 8%, and diabetic medication was changed**

### How the 24 Features Were Used to Define Groups 3 and 4

#### To Identify Group 4 ("High, changed")

- The **A1c test result** column must be **>8**
- The **Change of medications** column (derived as described above) must be **change**

#### To Identify Group 3 ("High, not changed")

- The **A1c test result** column must be **>8**
- The **Change of medications** column must be **no change**
