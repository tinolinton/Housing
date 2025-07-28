# Property Price Prediction Deep Learning Model

## Overview

This repository contains a deep learning solution for predicting property prices using a real estate dataset. The workflow includes exploratory data analysis (EDA), data cleaning, feature engineering, preprocessing, model building, training, evaluation, and visualization. The model is built with TensorFlow/Keras and leverages hyperparameter tuning and advanced feature selection.

## Dataset

- **Source**: `Housing.csv`
- **Features**:
  - `price`: Target variable (property price)
  - `area`: Property area (square feet)
  - `bedrooms`: Number of bedrooms
  - `bathrooms`: Number of bathrooms
  - `stories`: Number of stories
  - `mainroad`: Proximity to main road (yes/no)
  - `guestroom`: Guest room availability (yes/no)
  - `basement`: Basement presence (yes/no)
  - `hotwaterheating`: Hot water heating (yes/no)
  - `airconditioning`: Air conditioning (yes/no)
  - `parking`: Number of parking spaces
  - `prefarea`: Preferred area (yes/no)
  - `furnishingstatus`: Furnishing status (furnished, semi-furnished, unfurnished)

## Workflow

### 1. Exploratory Data Analysis (EDA)

- Inspected data structure and missing values.
- Visualized price distribution and feature relationships.
- Identified outliers and skewed features.
- Key findings:
  - Price and area are right-skewed.
  - `area`, `bedrooms`, `bathrooms`, `stories`, and `parking` are most correlated with price.

### 2. Data Cleaning & Preprocessing

- Removed extreme values for `bedrooms` and `area`.
- Imputed missing values:
  - Median for numerical (`parking`)
  - Mode for categorical (`mainroad`, `hotwaterheating`, `prefarea`)
- Feature engineering:
  - Created interaction terms (`area_x_stories`, `bed_bath_ratio`)
  - New features: `price_per_area`, `total_rooms`, `luxury_score`
- Encoded categorical variables:
  - Binary mapping for yes/no columns
  - One-hot encoding for `furnishingstatus`
- Outlier handling:
  - Winsorization for `price` and `area`
- Skewness correction:
  - PowerTransformer applied to skewed features

### 3. Feature Selection

- Used mutual information to select top predictive features.

### 4. Model Building

- Multilayer Perceptron (MLP) with:
  - Input layer (based on selected features)
  - 3 Dense layers (ReLU activation)
  - Batch Normalization and Dropout for regularization
  - Output layer (linear activation)
- Adam optimizer, MSE loss, and early stopping.

### 5. Training

- Data split: 80% train, 20% test.
- Trained for up to 300 epochs with batch size 32.
- Early stopping to prevent overfitting.

### 6. Evaluation

- **Metrics Used:**
  - **Mean Absolute Error (MAE):** Measures the average magnitude of errors between predicted and actual prices, without considering direction. Lower values indicate better accuracy.
  - **Mean Squared Error (MSE):** Calculates the average of squared differences between predicted and actual prices. Penalizes larger errors more than MAE.
  - **Root Mean Squared Error (RMSE):** The square root of MSE, representing error in the same units as the target variable. Easier to interpret than MSE.
  - **R² (Coefficient of Determination):** Indicates the proportion of variance in the actual prices explained by the model. Values closer to 1 mean better fit.

- **Results:**
  - `Test MAE: ~0.18`
  - `Test MSE: ~0.058`
  - `Test RMSE: ~0.24`
  - `Test R²: ~0.95`

### 7. Visualizations

- Training history (loss curves)
- Residual analysis
- Feature importance (from first layer weights)
- Actual vs Predicted scatter plot
- Error distribution histogram

## Key Techniques

- Mutual Information for feature selection
- PowerTransformer for normalization
- Early Stopping for regularization
- Batch Normalization for stable training

## Dependencies

- Python 3.7+
- Libraries:
  ```requirements
  pandas==1.3.4
  numpy==1.21.4
  matplotlib==3.5.1
  seaborn==0.11.2
  scikit-learn==1.0.2
  tensorflow==2.7.0
  scikeras==0.7.0
  jupyter==1.0.0
  ```

## Results Visualization

![Prediction Visualization](prediction_visualization.png)
_Figure: Actual vs Predicted prices show strong linear relationship (R² ≈ 0.95)_

## Usage

1. Clone the repository and install dependencies.
2. Run [code_baseModel.ipynb](code_baseModel.ipynb) in Jupyter Notebook.
3. Inspect outputs and visualizations in the notebook.
