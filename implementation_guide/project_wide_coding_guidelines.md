# Project-Wide Coding Guidelines for Agentic AI Coder

## Project Overview

**Objective**: To build, train, and evaluate a neural network model for predicting 30-day hospital readmission in diabetes patients.

**Core Principle**: This project will be executed in a modular, pipeline-based fashion. Each major phase (Data Exploration, Feature Engineering, Model Building, etc.) will be contained in its own Jupyter Notebook (.ipynb). The output of one phase (e.g., a cleaned CSV file) becomes the input for the next. This ensures clarity, reproducibility, and ease of debugging.

## Notebook Workflow and Naming Convention

The project is a linear pipeline. Each notebook must follow this sequence:

### 1_data_exploration_cleaning.ipynb

- **Input**: data/raw/diabetic_data.csv
- **Tasks**: Load data, perform initial exploration, handle missing values, clean data types, apply initial transformations
- **Output**: data/processed/1_cleaned_data.csv

### 2_feature_engineering.ipynb

- **Input**: data/processed/1_cleaned_data.csv
- **Tasks**: Encode categorical variables, scale numerical variables, create interaction terms, select final features
- **Output**: data/processed/2_featured_data.csv

### 3_model_training_evaluation.ipynb

- **Input**: data/processed/2_featured_data.csv
- **Tasks**: Split data into train/test sets, build and train baseline MLP, apply regularization techniques (dropout, batch norm, early stopping), evaluate all models, save best model
- **Outputs**: models/baseline_mlp_model.h5, performance metrics/plots

### 4_final_report_generation.ipynb (Optional but recommended)

- **Input**: Outputs from previous notebooks, especially 3_model_training_evaluation.ipynb
- **Tasks**: Consolidate all figures, tables, and key results into a final, clean notebook that can be easily converted to PDF for submission

## Coding Standards and Best Practices

### 3.1. Environment Setup and Imports

At the top of every notebook, include a "Setup" cell.

```python
# ==============================================================================
# SETUP CELL: Environment and Imports
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Set project root directory for robust path handling
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("Libraries imported and environment set successfully.")
print(f"Project Root: {PROJECT_ROOT}")
```

### 3.2. File I/O (Input/Output)

- **Always use os.path.join()** to construct file paths. This makes code OS-independent
- **Never hardcode absolute paths**. Use PROJECT_ROOT and other directory variables defined in setup cell

**Example for Loading Data:**

```python
# In notebook 1
file_path = os.path.join(RAW_DATA_DIR, 'diabetic_data.csv')
df = pd.read_csv(file_path)
```

**Example for Saving Data:**

```python
# At the end of notebook 1
output_path = os.path.join(PROCESSED_DATA_DIR, '1_cleaned_data.csv')
df.to_csv(output_path, index=False)
print(f"Cleaned data saved to {output_path}")
```

### 3.3. Code Readability and Documentation

- **Markdown Cells**: Use Markdown cells extensively to explain the purpose of each code block. Structure your notebook like a report
- **Use #** for main section titles (e.g., ## 1. Handling Missing Values)
- **Use * or -** for sub-points
- **Use **bold** for emphasis on key actions or findings
- **Comments**: Add inline comments (#) to explain complex or non-obvious lines of code
- **Variable Names**: Use descriptive variable names (e.g., df_encoded instead of df2, y_train_resampled instead of ytr)

### 3.4. Reproducibility

- **Random States**: For any operation involving randomness (e.g., train_test_split, SMOTE, model initialization), always set random_state=42. This ensures that you and anyone else running the code gets the exact same results

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3.5. Modularity and Function Writing

- For repeated tasks (e.g., calculating missing value percentages, plotting a confusion matrix), encapsulate the logic in a function
- **Define all functions at the top of the notebook** or in a dedicated "Functions" section

**Example Function:**

```python
def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Plots a confusion matrix using Seaborn."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
```

## Notebook-Specific Instructions

### Notebook 1: 1_data_exploration_cleaning.ipynb

**Goal**: Produce 1_cleaned_data.csv

**Key Actions:**

- Load diabetic_data.csv
- Replace '?' with np.nan
- Systematically analyze and document missing value percentages
- Perform data cleaning as per project plan (e.g., drop weight, payer_code, handle medical_specialty)
- Perform initial type corrections (e.g., convert age categories)
- Save cleaned DataFrame to data/processed/1_cleaned_data.csv without index

### Notebook 2: 2_feature_engineering.ipynb

**Goal**: Produce 2_featured_data.csv, ready for modeling

**Key Actions:**

- Load 1_cleaned_data.csv
- Define features (X) and target (y). The target is readmitted, which should be converted to a binary format (e.g., <30 -> 1, >30 or NO -> 0)
- Perform categorical encoding (One-Hot Encoding)
- Perform numerical scaling (StandardScaler)
- **Crucially**: Save the fitted StandardScaler object using joblib.dump() so it can be reused on test set or new data. Save it in the models/ directory
- Save final, fully featured DataFrame to data/processed/2_featured_data.csv

### Notebook 3: 3_model_training_evaluation.ipynb

**Goal**: Train, evaluate, and save the best model

**Key Actions:**

- Load 2_featured_data.csv
- Split into X_train, X_test, y_train, y_test
- **Baseline Model**: Build, compile, and train a simple MLP. Evaluate its performance (accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix)
- **Regularized Models**: Systematically add:
  - **Dropout**: Add Dropout layers
  - **Batch Normalization**: Add BatchNormalization layers
  - **Early Stopping**: Use EarlyStopping callback
- For each model variant, log its performance in a summary table for easy comparison
- Save the best-performing model (e.g., using model.save('models/best_mlp_model.h5'))
- Generate all necessary plots for your report (e.g., training history, ROC curves)

## Final Deliverable Generation

The final output for your assessment is a PDF report. This should be generated from a clean, well-structured notebook (or a Word document) that pulls in key findings, tables, and figures generated throughout the pipeline. The 4_final_report_generation.ipynb notebook can be used to assemble these final artifacts before exporting to PDF.
