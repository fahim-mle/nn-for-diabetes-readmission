# Coding Agent Guide: Notebook 2 - Feature Engineering

## Objective

This notebook will transform the cleaned dataset (`1_cleaned_data.csv`) into a format suitable for training a neural network. The primary tasks include defining the target variable, performing domain-specific feature engineering, applying One-Hot Encoding to categorical variables, scaling numerical features, and creating the final, modeling-ready dataset.

## Setup Instructions

At the beginning of the notebook, run the standard setup cell to configure the environment and paths. This will include importing necessary libraries like pandas, numpy, sklearn.preprocessing, and sklearn.model_selection.

```python
# ==============================================================================
# SETUP CELL: Environment and Imports
# ==============================================================================

# (Include standard imports: pandas, numpy, os, warnings)

# Add specific imports for this notebook
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib  # For saving the scaler
```

## Step 1: Load the Cleaned Data

**Action:**

- Load the `1_cleaned_data.csv` file from the `data/processed/` directory into a pandas DataFrame.
- Display the first few rows and the `.info()` summary to confirm the data has loaded correctly.

**Expected Output:**

- A DataFrame named `df_clean`.
- Printouts of the `.head()` and `.info()` methods.

**Markdown Note:** Add a brief markdown note confirming that the cleaned data has been successfully loaded and is ready for feature engineering.

## Step 2: Perform Domain-Specific Feature Engineering

**Action:**
This is a critical step where we create new, more meaningful features from existing ones, based on the logic described in the research paper.

### Create the `change_of_meds` Feature

**Logic:** A "change" is defined as any modification to the treatment plan. For each patient encounter, scan the 24 medication columns. If any of these columns has the value 'up' or 'down', it signifies a change. The new feature, `change_of_meds`, should be 1 for change and 0 for no change.

**Implementation:** Create a boolean mask for all 24 medication columns where the value is 'up' or 'down'. Use `.any(axis=1)` on this mask to create a boolean Series, then convert it to an integer (1/0).

### Create the Key `HbA1c_Change_Group` Feature

**Logic:** This feature combines the HbA1c test result with the medication change to model the paper's central hypothesis. Create a new categorical feature with four distinct groups:

- **Group 1:** 'No Test' - A1Cresult is 'None'.
- **Group 2:** 'Normal' - A1Cresult is 'Norm'.
- **Group 3:** 'High, No Change' - A1Cresult is '>7' or '>8' AND change_of_meds is 0.
- **Group 4:** 'High, Change' - A1Cresult is '>7' or '>8' AND change_of_meds is 1.

**Implementation:** Use `np.select` or a series of `.loc` assignments to create this new column.

**Expected Output:**

- Two new columns in `df_clean`: `change_of_meds` (binary 0/1) and `HbA1c_Change_Group` (categorical with 4 levels).
- Print the `value_counts()` for the new `HbA1c_Change_Group` feature.

**Markdown Note:** Add a detailed markdown section titled "Domain-Specific Feature Engineering" explaining the logic behind creating these new features.

## Step 3: Define Target and Feature Variables

**Action:**

### Create the Target Variable (y)

- Isolate the `readmitted` column and convert it to binary: `<30` -> 1, and `>30` or `NO` -> 0.
- The final target should be a pandas Series named `y`.

### Create the Feature Matrix (X)

- Create the feature matrix `X` by dropping the original `readmitted` column.
- Crucially, also drop the 24 individual medication columns and the original `change` column, as their information is now captured in the new, more powerful engineered features.

**Expected Output:**

- A pandas Series `y` and a DataFrame `X`.
- Print the `value_counts()` of `y` and the columns of `X` to confirm the structure is correct.

## Step 4: Identify Numerical and Categorical Features

**Action:**

- Inspect the data types of the columns in the new `X`.
- Create two separate lists of column names:
  - `numerical_features`: All columns that are inherently numerical (e.g., `time_in_hospital`, `num_lab_procedures`, `change_of_meds`).
  - `categorical_features`: All columns that are categorical (e.g., `race`, `gender`, `age`, `admission_type_id`, `medical_specialty`, `diag_1`, and your new `HbA1c_Change_Group`).

**Expected Output:**

- Two Python lists: `numerical_features` and `categorical_features`.
- Print these lists to verify they are correct.

**Markdown Note:** Add a markdown section listing the identified numerical and categorical features.

## Step 5: Split Data into Training and Testing Sets

**Action:**

- Use `train_test_split` to partition the data.
- Split `X` and `y` into training and testing sets using an 80/20 split (`test_size=0.2`).
- Use `stratify=y` to maintain class distribution.
- Set `random_state=42` for reproducibility.
- Name the resulting variables: `X_train`, `X_test`, `y_train`, `y_test`.

**Expected Output:**

- Four variables: `X_train`, `X_test`, `y_train`, `y_test`.
- Print the shapes and class distributions to confirm the split was successful.

## Step 6: Build a Preprocessing Pipeline with One-Hot Encoding

**Action:**
To prevent data leakage and handle different data types correctly, we will use a ColumnTransformer. This is the standard and most robust way to apply different transformations to different columns.

### Create a preprocessor object using ColumnTransformer

- **For `numerical_features`:** Create a pipeline that applies `StandardScaler()`. This scales the data to have a mean of 0 and a standard deviation of 1.
- **For `categorical_features`:** Create a pipeline that applies `OneHotEncoder()`.

### Why One-Hot Encoding?

Neural networks require all input to be numerical. One-Hot Encoding converts each category in a categorical feature into a new binary (0/1) column. This prevents the model from incorrectly assuming an ordinal relationship (e.g., that 'race_Caucasian' is greater than 'race_AfricanAmerican').

### Parameters

- Use `handle_unknown='ignore'` to prevent errors if a new category appears in the test set that wasn't in the training set.
- Use `sparse=False` to get a dense numpy array output, which is often easier to work with.

### Fit the preprocessor

Call the `.fit()` method on the preprocessor object, passing only the `X_train` data. This is critical. The scaler learns the mean/std from the training data, and the encoder learns the categories from the training data.

**Expected Output:**

- A fitted preprocessor object.

**Markdown Note:** Add a markdown section titled "Preprocessing Pipeline and One-Hot Encoding" that explains why One-Hot Encoding is necessary for neural networks and why the ColumnTransformer is the best practice for applying it alongside other transformations.

## Step 7: Transform the Data

**Action:**

- Use the fitted preprocessor to transform both the training and testing data.

  ```python
  X_train_transformed = preprocessor.transform(X_train)
  X_test_transformed = preprocessor.transform(X_test)
  ```

- The output will be numpy arrays. The number of columns will be much higher than before due to one-hot encoding.

**Expected Output:**

- Two numpy arrays: `X_train_transformed` and `X_test_transformed`.
- Print the shapes of these new arrays.

## Step 8: Save the Preprocessed Data and the Scaler

**Action:**

### Save the Preprocessor

- Use `joblib.dump()` to save the preprocessor object to the `models/` directory.
- Name it `preprocessor.joblib`.

### Save the Final Data

- Use `np.savez_compressed()` to save all four final data objects (`X_train_transformed`, `X_test_transformed`, `y_train`, `y_test`) into a single file in the `data/processed/` directory.
- Name this file `3_final_data.npz`.

**Expected Output:**

- A file named `preprocessor.joblib` in the `models/` directory.
- A file named `3_final_data.npz` in the `data/processed/` directory.
- Print confirmation messages showing the exact paths where the files were saved.
