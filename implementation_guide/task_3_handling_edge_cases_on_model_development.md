# **Coding Agent Guide: Handling Edge Cases and Unexpected Issues**

## **Objective**

This guide provides instructions for your coding agent to follow when encountering unexpected errors, data anomalies, or performance issues during the model training and evaluation phase (Notebook 3). The goal is to build resilience into the code and ensure it can handle problems gracefully.

---

## **Edge Case 1: Training is Extremely Slow**

### **Problem:**EC1

The `.fit()` process is taking an unusually long time to complete each epoch, much longer than expected.

#### **Diagnostic Questions:**EC1

* Are you running on a **CPU** instead of a **GPU**? A CPU is significantly slower for the matrix calculations in neural networks.
* Is the **batch size** too small or too large? A very small batch size can be inefficient, while a very large one can cause memory issues.
* Is the **model architecture** too complex? More layers and neurons mean more calculations.
* Are you running other **resource-intensive programs** at the same time?

#### **Action Plan:**

1. **Check for GPU:** In a new cell, run the following code to confirm if TensorFlow is detecting a GPU.

    ```python
    from tensorflow.python.client import device_lib
    print("Available devices:", device_lib.list_local_devices())
    ```

2. **Optimize Batch Size:** If on a CPU, a good starting point is 32 or 64. If you have a lot of RAM, you might need to decrease it. If on a GPU, you can use a much larger batch size (e.g., 128, 256).
3. **Reduce Model Complexity (if needed):** If training is still too slow, try temporarily reducing the number of neurons or layers to see if it's an architectural problem.
4. **Use `verbose=0`:** Set `verbose=0` in the `.fit()` call to stop printing progress for each epoch, which can speed things up slightly.

---

## **Edge Case 2: Model Performance is Extremely Poor (e.g., ~50% Accuracy)**

### **Problem:**EC2

The model is not learning, with training and validation loss staying high and not improving.

#### **Diagnostic Questions:**EC2

* **Is the learning rate too high or too low?** A bad learning rate can cause the model to overshoot minima or converge too slowly.
* **Is the data not scaled correctly?** If the mean/variance of the training and test sets are very different, the model will struggle.
* **Is the data shuffled incorrectly?** If the validation set has a very different distribution of classes than the training set, performance will be poor.
* **Is the loss function inappropriate?** `binary_crossentropy` is correct for classification, but ensure you didn't accidentally use a regression loss like `mse`.

#### **Action Plan:**EC2

1. **Verify Data Scaling:** Load the preprocessor and check its `mean_` and `var_` attributes. They should be non-zero and reasonable.
2. **Check Class Distribution:** Print `y_train.value_counts(normalize=True)` to ensure classes are balanced. A severe imbalance can make accuracy a misleading metric.
3. **Simplify the Model:** Temporarily switch back to the baseline model (128-64-1) without any regularization to see if the problem is with the added complexity.
4. **Adjust Learning Rate:** In the `model.compile()` step, try a different optimizer or a specific learning rate (e.g., `Adam(learning_rate=0.001)`).
5. **Run for More Epochs:** The model might just need more time to converge. Increase `epochs` to 100 or 200.

---

## **Edge Case 3: "ValueError: Input 0 is incompatible..."**

### **Problem:**

This error almost always means there is a **mismatch between the number of features the model expects and the number of features it receives**. This typically happens after a change in the feature engineering step.

#### **Diagnostic Questions:**

* Did you **change the number of features** in Notebook 2 after you already started Notebook 3?
* Did you **re-save the preprocessed data** (`3_final_data.npz`) after making changes?
* Are you loading the **correct preprocessor** (`preprocessor.joblib`) that corresponds to the data you are loading?

#### **Action Plan:**EC3

1. **Check Shapes:** In Notebook 3, print the `shape` of `X_train_transformed` and compare it to the `input_dim` of your model.

    ```python
    print("Shape of X_train_transformed:", X_train_transformed.shape)
    print("Expected input_dim for model:", X_train_transformed.shape[1])
    ```

2. **Re-run the Pipeline:** If there is a mismatch, you **must** go back to Notebook 2, fix the feature engineering, and re-run the entire preprocessing pipeline (Step 6 in Notebook 2).
3. **Re-save Everything:** Re-save the `.npz` file and the `.joblib` file.
4. **Restart Kernel:** In Jupyter, go to the "Kernel" menu and select "Restart & Clear Output". Then re-run all cells in Notebook 3 from the beginning.

---

## **Edge Case 4: Overfitting is Severe**

### **Problem:**EC4

The training accuracy continues to improve towards 100%, but the validation accuracy gets worse or stays flat. The training loss goes down, but the validation loss starts to rise.

#### **Diagnostic Questions:**EC4

* **Is Dropout rate too low?** A rate of 0.1 or 0.2 might not be enough regularization.
* **Is the model too big?** Too many parameters for the amount of data.
* **Is Early Stopping not working?** The `monitor` might be set to the wrong metric (e.g., `'val_accuracy'` instead of `'val_loss'`).

#### **Action Plan:**EC4

1. **Increase Dropout:** Change the dropout rate from `0.3` to `0.5` in your regularized model.
2. **Add L2 Regularization:** Add `kernel_regularizer=l2(0.01)` to your `Dense` layers. This penalizes large weights and is a powerful technique against overfitting.
3. **Add More Data Augmentation:** If possible, use techniques like SMOTE to create more balanced or varied training data.
4. **Check Early Stopping:** Ensure the `EarlyStopping` callback is configured correctly and is actually being used in the `.fit()` call.

---

## **General Debugging and Logging Strategy**

### **Action Plan:**Gen_debug

1. **Use `tf.keras.utils.plot_model`:** If you are getting strange errors, create a plot of your model architecture to visually inspect the layers and shapes.
2. **Wrap Code in `try...except`:** For any non-critical code that might fail (like loading files), wrap it in a `try...except` block to prevent the entire notebook from crashing.

    ```python
    try:
        model = load_model('my_model.h5')
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    ```

3. **Print Everything:** Use print statements or logging at the end of each major step (e.g., after `.fit()`, after `.evaluate()`) to create a paper trail of what the code did.
4. **Save Intermediates:** If a step is computationally expensive (like training a complex model), save the results (e.g., the trained model, the history object) to disk immediately so you don't have to re-run it.

By preparing for these edge cases, your coding agent will be more robust and able to complete the project without getting stuck, providing you with a much smoother experience.
