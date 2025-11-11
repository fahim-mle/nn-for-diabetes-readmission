# Coding Agent Guide: Notebook 3 - Model Training and Evaluation

## Objective

This notebook will build, train, and evaluate baseline Multiple Layer Perceptron (MLP) model. It will then systematically apply regularization techniques (Dropout, Batch Normalization, Early Stopping) to improve upon the baseline and compare their performance. The final output will be a trained model and a comprehensive performance analysis.

## Setup Instructions

At the beginning of the notebook, run the standard setup cell. This will now include imports for tensorflow/keras for building the neural network, as well as sklearn.metrics for evaluation.

```python
# ==============================================================================
# SETUP CELL: Environment and Imports
# ==============================================================================

# (Include standard imports: pandas, numpy, os, warnings)

# Add specific imports for this notebook
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Libraries imported and environment set successfully.")
```

## Step 1: Load the Preprocessed Data and Preprocessor

**Action:**

- Load the preprocessed data arrays from the `3_final_data.npz` file located in the `data/processed/` directory.
- Load the fitted preprocessor object from the `preprocessor.joblib` file located in the `models/` directory.
- Unpack the `.npz` file into the four variables: `X_train`, `X_test`, `y_train`, `y_test`.

**Expected Output:**

- Four numpy arrays: `X_train`, `X_test`, `y_train`, `y_test`.
- A fitted preprocessor object.
- Print the shapes of all loaded data arrays to confirm they are correct.

**Markdown Note:** Add a markdown section confirming that the preprocessed data and fitted preprocessor have been successfully loaded for modeling.

## Step 2: Build the Baseline MLP Model

**Action:**

- Get the number of input features from `X_train.shape[1]`.
- Use `tf.keras.models.Sequential` API to build the model as discussed.

**Architecture:**

- **Input Layer:** Dense layer with 128 neurons, `activation='relu'`, and `input_dim` set to the number of features.
- **Hidden Layer:** Dense layer with 64 neurons, `activation='relu'`.
- **Output Layer:** Dense layer with 1 neuron, `activation='sigmoid'`.

**Compile the Model:**

- **Optimizer:** Use `adam` (a robust, standard choice).
- **Loss Function:** Use `binary_crossentropy` (the standard for binary classification).
- **Metrics:** Include `['accuracy']` for easy monitoring during training.
- Print `model.summary()` to inspect the architecture, number of parameters, etc.

**Expected Output:**

- A compiled Keras Sequential model named `baseline_model`.
- A printed summary of the model's layers and parameters.

**Markdown Note:** Add a detailed section titled "Baseline Model Architecture" that explains the chosen architecture (128-64-1), the choice of activation functions (ReLU, Sigmoid), and the compilation parameters (Adam optimizer, Binary Crossentropy loss), justifying each choice as discussed previously.

## Step 3: Train and Evaluate the Baseline Model

**Action:**

### Train the Model

- Call the `.fit()` method on `baseline_model`.
- Pass `X_train`, `y_train` as training data.
- Use `validation_data=(X_test, y_test)` to monitor performance on unseen data during training.
- Set `epochs=50` and `batch_size=32`.
- Set `verbose=1` to see the training progress.
- Store the training history in a variable called `history_baseline`.

### Evaluate Performance

- Use the trained model to make predictions on the test set: `y_pred_proba = baseline_model.predict(X_test)`.
- Convert probabilities to binary class predictions (0 or 1) using a 0.5 threshold: `y_pred_class = (y_pred_proba > 0.5).astype(int)`.
- Calculate and print the following metrics:
  - **Classification Report:** `classification_report(y_test, y_pred_class)`
  - **Confusion Matrix:** `confusion_matrix(y_test, y_pred_class)`
  - **ROC AUC Score:** `roc_auc_score(y_test, y_pred_proba)`
- Plot the training history (accuracy and loss curves) using matplotlib.

**Expected Output:**

- A trained `baseline_model`.
- A `history_baseline` object.
- Printed classification report, confusion matrix, and ROC AUC score.
- Two plots: one for training/validation accuracy, one for training/validation loss.

**Markdown Note:** Create a section "Baseline Model Evaluation" where you present these results. Add a brief interpretation of metrics (e.g., "The baseline model achieved an accuracy of X%, with a ROC AUC of Y%. The confusion matrix shows...").

## Step 4: Apply Regularization Techniques

Now, you will build new models, each adding one regularization technique, to compare their impact against the baseline.

### Action 4a: Model with Dropout

- **Build a new model:** Create a new Sequential model, identical to the baseline.
- **Add Dropout:** After the first `Dense(128, ...)` layer and after the second `Dense(64, ...)` layer, add a `Dropout(0.3)` layer. A dropout rate of 0.3 is a good starting point.
- Compile this model with the same settings as the baseline.
- Train and evaluate this model exactly as in Step 3, saving the history and results with new names (e.g., `history_dropout`, `dropout_model`).

### Action 4b: Model with Batch Normalization

- **Build a new model:** Create a new Sequential model.
- **Add Batch Normalization:** Add a `BatchNormalization()` layer before the activation function of each dense layer. A common pattern is Dense -> BatchNorm -> Activation.
- Compile, train, and evaluate this model exactly as before, saving results with new names (e.g., `history_batchnorm`, `batchnorm_model`).

### Action 4c: Model with Early Stopping

- Use the baseline model architecture. No changes to the model structure are needed.
- **Create an EarlyStopping Callback:**

  ```python
  early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
  ```

  This will stop training if the validation loss doesn't improve for 5 consecutive epochs and will restore the weights from the best-performing epoch.
- **Train the model:** Call `.fit()` as before, but this time add the callback: `callbacks=[early_stop]`.
- Evaluate the resulting model.

**Expected Output for Step 4:**

- Three new trained models and their corresponding history objects.
- A set of evaluation metrics and plots for each of the three regularized models.

**Markdown Note:** For each technique, create a subsection (e.g., "4a. Impact of Dropout"). Present the results and write a brief analysis comparing them to the baseline. For example, "Adding Dropout reduced overfitting, as seen by the smaller gap between training and validation loss."

## Step 5: Compare All Models and Save the Best

**Action:**

### Create a Summary Table

- Create a pandas DataFrame to summarize key performance metrics (e.g., Test Accuracy, Test ROC AUC) for all four models (Baseline, Dropout, BatchNorm, EarlyStopping).

### Visual Comparison

- Create a single plot with ROC curves for all four models on the same axes to visually compare their performance.

### Select the Best Model

- Based on the summary table and plots, decide which model performed best. Typically, this is the one with the highest ROC AUC and good generalization (low overfitting).

### Save the Best Model

- Use `best_model.save('models/best_mlp_model.h5')` to save the chosen model's architecture and weights to the `models/` directory.

**Expected Output:**

- A summary table comparing the performance of all models.
- A combined ROC curve plot.
- A saved `.h5` file for the best model.

**Markdown Note:** Create a final section "Model Comparison and Selection". Present the summary table and ROC curve plot. Write a concluding paragraph justifying which model you selected as the best and why.
