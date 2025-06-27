
# AI Analysis on Massive Bank Dataset using GPU Acceleration

## Introduction

This project applies AI and Machine Learning techniques to the Massive Bank Dataset, a large-scale transaction dataset. The primary goals are:

1.  To analyze transaction flow patterns, potentially identifying abnormal behavior across different business domains and locations using Exploratory Data Analysis (EDA).
2.  To build and compare supervised learning models (Random Forest, XGBoost, and Linear Regression) for predicting the average transaction value (`avg_transaction_value`).
3.  To leverage GPU acceleration using RAPIDS (cuDF, cuML) and XGBoost for efficient processing and model training on this large dataset (>1 million records) and evaluate resource usage alongside model performance.

The problem involves understanding patterns and making predictions (supervised learning).

## Dataset

*   **Name:** Massive Bank Dataset
*   **Source:** Kaggle (Originally mentioned, though the file loaded is `bankdataset.xlsx`)
*   **Size:** Over 1 million records (1,004,480 rows, 5 initial columns).
*   **Description:** Provides daily summaries of bank transactions across various domains and locations.

**Columns:**
*   `Date`: The day when the transactions occurred.
*   `Domain`: Type of business where the transaction happened (e.g., RETAIL, INVESTMENTS, MEDICAL, etc.).
*   `Location`: City or region of the transaction.
*   `Value`: Total value (in currency) of all transactions on that day for the specific Domain/Location.
*   `Transaction_count`: Total number of individual transactions on that day for the specific Domain/Location.

## Methodology

### Environment & Libraries
The analysis was performed using Python on a system equipped with an NVIDIA T4 GPU. Key libraries employed include:
*   **Data Manipulation:** Pandas, cuDF (for GPU-accelerated dataframes)
*   **Numerical Computation:** NumPy, CuPy (for GPU-accelerated arrays)
*   **Machine Learning:**
    *   cuML (RAPIDS library for GPU-accelerated ML: `train_test_split`, `LinearRegression`, `RandomForestRegressor`)
    *   XGBoost (GPU-enabled version)
    *   Scikit-learn (for `train_test_split` in one instance and metrics)
*   **Visualization:** Matplotlib, Seaborn
*   **Utilities:** Joblib (for model saving), GPUtil (for monitoring GPU status)

### Data Loading and Preprocessing
1.  The dataset was loaded from an Excel file (`newbankdataset.xlsx`) into a pandas DataFrame and then converted to a cuDF DataFrame for GPU processing.
2.  Data types were checked, and the `Date` column was converted to datetime objects.
3.  Feature Engineering:
    *   `Year`, `Month`, `Day`, and `month_name` were extracted from the `Date` column.
    *   The target variable, `avg_transaction_value`, was calculated as `Value / Transaction_count`.
    *   Categorical features (`Domain`, `Location`) were one-hot encoded for modeling.
4.  A check confirmed there were no missing values in the dataset.
5.  Domain-specific multipliers were applied to the 'Value' column to potentially adjust for different business scales before recalculating `avg_transaction_value`.

### Exploratory Data Analysis (EDA)
Several visualizations were generated to understand patterns:
*   Mean transaction value by month for the top 5 locations.
*   Average transaction value trend over time for the 'RETAIL' domain.
*   Comparison of average daily transaction value across different domains.
*   Monthly trends in average daily transaction value across all domains.
*   Correlation heatmap between numeric features (`Value`, `Transaction_count`, `avg_transaction_value`).

### Modeling
The dataset (after preprocessing and encoding) was split into training and testing sets using both `cuml.model_selection.train_test_split` and `sklearn.model_selection.train_test_split`. Three regression models were trained to predict `avg_transaction_value`:

1.  **XGBoost:** An `XGBRegressor` model was trained (likely utilizing the GPU).
2.  **Linear Regression:** A `cuml.linear_model.LinearRegression` model was trained on the GPU.
3.  **Random Forest:** A `cuml.ensemble.RandomForestRegressor` model was trained on the GPU.


## Results

After performance evaluation between the three model, low scores indicate that the models, with the current features and preprocessing, struggled to accurately predict the `avg_transaction_value`.

**GPU Usage:**
The notebook confirms the use of an NVIDIA T4 GPU. Training times were recorded:
*   XGBoost fit: ~7.0s
*   Random Forest fit: ~9.8s (includes data conversion and split)

Final GPU status check showed moderate memory usage (~14.7%) and temperature (~77°C), indicating the GPU was utilized but not fully saturated by the final model state.

## Conclusion

This project demonstrated the process of analyzing a large bank transaction dataset using GPU acceleration with RAPIDS and XGBoost. EDA revealed some trends related to time and business domain. However, the predictive models (XGBoost, Linear Regression, Random Forest) built to estimate the average transaction value.

Despite the low predictive performance, the project successfully showcased the efficiency gains from using GPU-accelerated libraries (cuDF, cuML) for data manipulation and model training on a dataset exceeding 1 million rows.

## Lessons Learned

*   **Feature Importance:** Predicting derived metrics like `avg_transaction_value` can be challenging. The original `Value` and `Transaction_count`, along with time and categorical features, were not strong predictors in their raw or one-hot encoded form. More complex feature engineering (e.g., interaction terms, lagged features, domain-specific scaling beyond simple multipliers) might be necessary.

*   **GPU Acceleration Benefits:** Handling large datasets (>1M rows) was significantly faster using cuDF and cuML compared to traditional CPU-based pandas/sklearn, especially for tasks like data loading, transformation, and model training (as seen with cuML RF and XGBoost).
