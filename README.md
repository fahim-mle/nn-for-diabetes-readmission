# Neural Network for Diabetes Readmission Prediction

## Project Overview

This project implements a modular, pipeline-based approach to build, train, and evaluate a neural network model for predicting 30-day hospital readmission in diabetes patients. The project follows the methodology outlined in the research paper "Impact of HbA1c Measurement on Hospital Readmission Rates."

## Project Structure

```txt
assessment_main/
├── README.md                           # This file - project overview and setup
├── requirements.txt                     # Python dependencies
├── .gitignore                         # Git ignore file
├── data/                              # Data directory
│   ├── raw/                           # Raw data files
│   │   └── diabetic_data.csv        # Original dataset
│   └── processed/                      # Processed data files
│       ├── 1_cleaned_data.csv         # After exploration and cleaning
│       └── 2_featured_data.csv         # After feature engineering
├── notebooks/                          # Jupyter notebooks for pipeline
│   ├── 1_data_exploration_cleaning.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_model_training_evaluation.ipynb
│   └── 4_final_report_generation.ipynb
├── models/                             # Trained models and artifacts
│   ├── best_mlp_model.h5               # Best performing model
│   └── scaler.pkl                     # Fitted scaler object
├── utils/                              # Utility functions and scripts
├── figures/                            # Generated plots and visualizations
├── reports/                            # Final reports and exports
├── documentation/                       # Project documentation
│   └── essential_predictors.md
├── implementation_guide/                 # Implementation guides
│   ├── project_wide_coding_guidelines.md
│   └── task_1a.md
├── reference/                          # Reference materials
│   ├── data_description.md
│   ├── feature_engineering_guide.md
│   └── fearture_importance.md
└── assessment_resources/                 # Assessment PDFs and resources
    ├── Impact_of_HbA1c_Measurement_on_Hospital_Readmission_Rates_Analysis_of_70.pdf
    └── MA5852_Assignment_2_Brisbane.pdf
```

## Pipeline Workflow

The project follows a linear pipeline approach where each notebook processes the output of the previous one:

### 1. Data Exploration and Cleaning (`1_data_exploration_cleaning.ipynb`)

- **Input**: `data/raw/diabetic_data.csv`
- **Tasks**: Load data, perform initial exploration, handle missing values, clean data types
- **Output**: `data/processed/1_cleaned_data.csv`

### 2. Feature Engineering (`2_feature_engineering.ipynb`)

- **Input**: `data/processed/1_cleaned_data.csv`
- **Tasks**: Encode categorical variables, scale numerical variables, create interaction terms
- **Output**: `data/processed/2_featured_data.csv`

### 3. Model Training and Evaluation (`3_model_training_evaluation.ipynb`)

- **Input**: `data/processed/2_featured_data.csv`
- **Tasks**: Split data, build/train MLP, apply regularization, evaluate models
- **Output**: `models/best_mlp_model.h5`, performance metrics

### 4. Final Report Generation (`4_final_report_generation.ipynb`)

- **Input**: Outputs from previous notebooks
- **Tasks**: Consolidate figures, tables, and results
- **Output**: Final report ready for PDF export

## Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd assessment_main

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

- Place the `diabetic_data.csv` file in the `data/raw/` directory
- Ensure the dataset follows the expected format as described in `reference/data_description.md`

### 3. Running the Pipeline

- Open and execute notebooks in order: 1 → 2 → 3 → 4
- Each notebook will automatically create the necessary output directories
- Follow the coding guidelines in `implementation_guide/project_wide_coding_guidelines.md`

## Key Features

- **Modular Design**: Each phase is self-contained with clear inputs/outputs
- **Reproducibility**: Fixed random states (random_state=42) throughout
- **Path Management**: OS-independent path handling using `os.path.join()`
- **Documentation**: Extensive markdown documentation in each notebook
- **Version Control**: Git repository with appropriate .gitignore

## Dependencies

See `requirements.txt` for the complete list of required packages:

- Core ML: numpy, pandas, scikit-learn
- Deep Learning: tensorflow, keras
- Visualization: matplotlib, seaborn, plotly
- Development: jupyter, pytest, black

## Coding Standards

All code follows the guidelines in `implementation_guide/project_wide_coding_guidelines.md`:

- Consistent environment setup in every notebook
- Descriptive variable names
- Function-based modularity
- Comprehensive documentation
- Error handling and reproducibility

## Output Artifacts

- **Cleaned Data**: Processed dataset with handled missing values
- **Featured Data**: Engineered features ready for modeling
- **Trained Model**: Best performing neural network model
- **Performance Metrics**: Comprehensive evaluation results
- **Visualizations**: Plots for analysis and reporting
- **Final Report**: Consolidated findings ready for submission

## Contributing

1. Follow the established coding guidelines
2. Ensure reproducibility with fixed random states
3. Document all changes and decisions
4. Test thoroughly before committing changes
5. Use descriptive commit messages

## License

This project is developed for academic assessment purposes.
