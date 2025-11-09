# Task 1: Data Exploration and Cleaning Completion Instructions

This document provides detailed step-by-step instructions for completing the data exploration and cleaning phase of the diabetes readmission analysis project.

---

## Step 1: Load Initial Data and Perform Basic Inspection

### Action

- Load the `diabetic_data.csv` file from the `data/raw/` directory into a pandas DataFrame.
- Display the first 10 rows to get a visual sense of the data.
- Display the last 5 rows to check for any footer or summary rows.
- Use `.info()` to get a concise summary of the DataFrame, including column names, non-null counts, and data types.
- Use `.shape` to confirm the number of rows and columns.

### Expected Output

- A DataFrame named `df_raw`.
- Print statements showing the head, tail, info, and shape of the DataFrame.

### Markdown Note for step 1

Add a markdown cell summarizing the initial findings (e.g., "The dataset contains X rows and Y columns. Key observations include...").

---

## Step 2: Identify and Represent Missing Values

### Action step 2

- The dataset uses `'?'` to represent missing values. Replace all instances of `'?'` with `np.nan` (Not a Number) to enable proper handling with pandas functions.
- After replacement, calculate the number and percentage of missing values for each column.
- Sort the results by percentage in descending order to easily identify the most problematic columns.

### Expected Output step 2

- The DataFrame `df_raw` now has `np.nan` where `'?'` was.
- A printed table or Series showing the count and percentage of missing values for each column, sorted from highest to lowest.

### Markdown Note for step 2

Add a markdown cell titled "Missing Value Analysis" and explicitly state which columns have the highest missing data rates, quoting the percentages (e.g., "weight has 96.8% missing values, payer_code has 52.2%, etc."). This directly supports the exclusion rationale.

---

## Step 3: Handle Columns with Excessive Missing Data

### Action step 3

- Based on the analysis in Step 2 and the project plan, identify the columns to be removed due to excessive missing data: `weight`, `payer_code`, and `medical_specialty`.
- Drop the `weight` and `payer_code` columns from the DataFrame.
- For `medical_specialty`, do not drop it. Instead, replace its `np.nan` values with the string `'Missing'`. This preserves the variable while accounting for its missingness.
- Create a new DataFrame, `df_clean`, to hold the result of these operations.

### Expected Output step 3

- A new DataFrame `df_clean` with `weight` and `payer_code` removed.
- The `medical_specialty` column in `df_clean` should have no missing values, with `'Missing'` as a category.
- Print the new shape of `df_clean` and a confirmation message (e.g., "Dropped columns 'weight' and 'payer_code'. Imputed 'medical_specialty' with 'Missing' category.").

---

## Step 4: Analyze and Clean Individual Columns

### Action for step 4

Go through key columns one by one to inspect and clean their values.

#### **race:**

- Check the unique value counts (`.value_counts()`).
- Combine `?` (if any remain) and `Other` into a single category.

#### **gender:**

- Check the unique value counts.
- Investigate the `Unknown/Invalid` category. Count how many entries fall into it.
- Since it's a very small number, these rows can be dropped.

#### **age:**

- Check the unique value counts. Note that it's an ordered categorical variable (e.g., `[0-10)`, `[10-20)`).

#### **diag_1, diag_2, diag_3:**

- Check for missing values (`np.nan`).
- The original paper grouped these into broader categories. For this cleaning step, we will just ensure they are clean strings. For any remaining missing values in `diag_2` or `diag_3`, replace them with `'0'` (representing no secondary diagnosis).

#### **change** and **diabetesMed:**

- Check their value counts. These should be simple `'Ch'/'No'` and `'Yes'/'No'` categories. Ensure consistency.

### Expected Output step 4

- For each column analyzed, print the `value_counts()` before and after cleaning.
- The `df_clean` DataFrame should be updated with these changes.

### Markdown Note step 4

Document any decisions made, such as dropping rows with invalid gender or imputing diagnosis codes.

---

## Step 5: Remove Identifier Columns

### Action step 5

- Identify the columns that are unique identifiers: `encounter_id` and `patient_nbr`.
- Drop these two columns from `df_clean` as they have no predictive value and would cause overfitting.

### Expected Output step 5

- The `df_clean` DataFrame no longer contains `encounter_id` or `patient_nbr`.
- A print statement confirming the removal of these columns.
- Print the final shape of the DataFrame.

---

## Step 6: Final Data Type Conversion and Verification

### Action step 6

- Review the data types of all columns in `df_clean` using `.info()`.
- Convert columns to their most appropriate types to save memory and ensure correctness. For example:
  - Categorical columns like `race`, `gender`, `age`, `admission_type_id`, etc., can be converted to the `pd.Categorical` dtype.
  - Numerical columns like `time_in_hospital`, `num_medications` should be `int` or `float`.
- Run `.info()` again to verify the changes.

### Expected Output step 6

- A printout of the data types before and after conversion.
- A final `.info()` summary showing a clean, correctly typed DataFrame.

---

## Step 7: Save the Cleaned Dataset

### Action step 7

- Define the output file path using `os.path.join(PROCESSED_DATA_DIR, '1_cleaned_data.csv')`.
- Save the final `df_clean` DataFrame to this path.
- Crucially, set `index=False` to avoid writing the DataFrame index as an unnecessary column in the CSV file.
- Print a confirmation message showing the exact path where the file was saved.

### Expected Output step 7

- A file named `1_cleaned_data.csv` created in the `data/processed/` directory.
- A final confirmation message, e.g., `Cleaned dataset successfully saved to /path/to/project/data/processed/1_cleaned_data.csv`.

---

## TODO LIST RECOMMENDED

When starting a new task, it is recommended to create a todo list.

## Benefits of creating a todo list now

- Clear roadmap for implementation
- Progress tracking throughout the task
- Nothing gets forgotten or missed
- Users can see, monitor, and edit the plan

## Example structure

```markdown
- [ ] Analyze requirements
- [ ] Set up necessary files
- [ ] Implement main functionality
- [ ] Handle edge cases
- [ ] Test the implementation
- [ ] Verify results
```

Keeping the todo list updated helps track progress and ensures nothing is missed.
