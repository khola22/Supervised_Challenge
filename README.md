# Supervised Learning on Musical Dataset

## Introduction
The goal of this project is to apply supervised learning techniques to a musical dataset to classify and analyze musical tracks. This repository contains the code and resources used to process the data and train the models.

N.B: all the functions are defined in the file `helper.py`

## Phase 1: Data Preparation (data_prep_final.ipynb)

The first phase of the project focused on cleaning, exploring, and transforming the raw data into a format suitable for machine learning algorithms. The process is detailed below.

### 1. Data Exploration
We began by exploring the available datasets to understand their structure and identify quality issues.
- **Datasets Analyzed**: `genres`, `echonest`, `tracks`, and `spectral`.
- **Key Observation**: Significant missing values were identified, particularly in the `top_genre` (approx. 55% missing), `artist_latitude`, and `artist_longitude` columns.

### 2. Handling Missing Values

#### `top_genre` Imputation
To address the high percentage of missing values in the target variable `top_genre`, we implemented a hierarchical imputation strategy:
1.  **ID Conversion**: We converted `top_genre` from titles to IDs using the `genres` dataset to standardize the data.
2.  **Rooted Parent Imputation**: We traversed the genre hierarchy upwards to find a root parent genre.
    - If a dominant root parent was found, it was used to fill the missing value.
    - If multiple root parents existed without a clear dominant one, a parent was selected randomly.
    - *Justification*: This approach maximizes data utility by inferring likely genres based on the dataset's inherent hierarchy.

#### Artist Location
Missing values in `artist_latitude` and `artist_longitude` (approx. 60%) were handled as follows:
- **Imputation**: Missing values were replaced with `0`.
- **Feature Engineering**: A new binary column, `artist_location_unknown`, was created (1 for unknown, 0 for known).
- *Justification*: This preserves the signal that the location information was originally missing, which can be a predictive feature in itself.

### 3. Data Merging
We merged the main `tracks` dataset with the `spectral` dataset.
- **Exclusion of Echonest**: The `echonest` dataset was excluded at this stage because merging it would have reduced the total dataset size by approximately 90%.
- *Note*: We reserve the option to test `echonest` features in future iterations if accuracy improvements are needed.

### 4. Feature Reduction and Engineering
To optimize model performance and reduce multicollinearity, we analyzed feature correlations:
- **Correlation Analysis**: We calculated a correlation matrix for all features.
- **High Correlation (>90%)**: For groups of features exhibiting very high correlation, we retained only the mean column and dropped the others to avoid redundancy.
- **Principal Component Analysis (PCA)**: For the remaining features, we applied PCA to reduce dimensionality while retaining the most significant variance in the data.

### 5. Final Output
The processed and cleaned dataset has been saved as:
- **File**: `data/tracks_spectral_reduced.csv`

## Task 1: Predict the original genre (genre_top) (task1.ipynb)

## Task 2: Predict your coarse-grained genre (3–4 categories) (task2.ipynb)

## Task 3: Predict the track duration (task3.ipynb)
**Data Preparation**:  
- Since we applied regression models in this part, we had to handle categorical columns so high cardinality columns (`artist_name`, `title`, `album_title`) were dropped.
- **Target Transformation:** The target variable, `duration`, was handled for extreme outliers by **clipping** at 2000s and then **log-transformed** (`log(1+x)`) to achieve a near-normal distribution for regression stability.
- **Final Feature Sets:** Two primary feature sets were used for modeling:
    1.  **Set A (Large):** Metadata + Reduced Spectral Features ($\approx 100,000$ tracks).
    2.  **Set C (Complete/Small):** Metadata + Reduced Spectral + **Echonest** features ($\approx 10,000$ tracks).

**Regression models:**  
The goal is to build the best regression model to predict the log-transformed track duration. We compare various models and feature sets using the $\mathbf{R^2}$ score and the $\mathbf{MAE}$ (Mean Absolute Error) converted back to seconds for interpretability.  

| Model | Feature Set | Data Size (Tracks) | R² Score | MAE (Seconds) |
| :--- | :--- | :--- | :--- | :--- |
| KNN Regressor | Set A (Spectral) | $\approx 100,000$ | 0.1635 | 0.62 s |
| **Random Forest** | **Set A (Spectral)** | $\approx 100,000$ | **0.3643** | **0.52 s** |
| Random Forest | Set C (Complete) | $\approx 10,000$ | 0.3288 | 0.52 s |
| Gradient Boosting | Set C (Complete) | $\approx 10,000$ | **0.3643** | **0.52 s** |

The results indicate a strong **Quantity vs. Quality Trade-off** :

* The **Random Forest** model achieves the best performance ($R^2=0.3643$) on the **largest dataset (Set A)**.
* Adding the high-quality **Echonest features** (Set C) but losing $90\%$ of the data **neutralizes** the performance, resulting in the same $R^2$. The data quantity currently outweighs the feature quality.
* The **Gradient Boosting Regressor** failed to extract additional signal compared to the Random Forest on the smaller dataset.