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
- The supervised learning was performed on the numerical features of the dataset.
- The dataset has 19 numerical features. None of them was dropped. (F-test scores and variances are not low)
- The target is encoded to insure it is compact.
- 80% of the dataset is dedicated to the training and 20% for the test (with the same class proportions).
- Standardization of the features for the models that require it.
- The training data is split to 5 folds with the same class distribution. In each iteration on fold becomes the validation set and the other four are used as a trining set.

  **Results (tracks_spectral dataset)**
  <img width="729" height="299" alt="image" src="https://github.com/user-attachments/assets/76319ccc-35fc-4186-b228-21651645c58c" />

The best model is **LightGBM** with an accuracy of 58% and an overfitting value within an acceptable range.
  
Before the final test :
- We try to find better hyperparameters to improve the best model's accuracy. That improves tha accuracy of 4.4%
- The model is trained again on the whole training/validation set
** Result **
  The final test result are : 65.44% accuracy
** Confusion matrix **
 <img width="708" height="422" alt="image" src="https://github.com/user-attachments/assets/5189d364-bb8e-415c-b0e5-2023e4b6f81e" />


- The diagonale is pretty dominant, which indicates a good class separation (65.44%)
- The best predictes classes are :
     - 11 (originally 38). 75% was well predicted
     - 7  (originally 15). 74% was well predicted
     - 5 (originally 12). 69% was well predicte
  This is explained by the fact that thes 3 classes represent over 67% of the dataset. The other 11 classes are a minority.
The class 12 (763) was always mistaken for class 11 (38), and this is predictable as class 12 represents only 0,005% of the data.
All in all, there's a strong correlation between the percentage of the class in the data and the likelihood to predict it right.

We can also notice that the classes are confused the most with the dominant classes (11,7,5)
The results are explained by the imbalanced data.


## Task 2: Predict your coarse-grained genre (3–4 categories) (task2.ipynb)

## Task 3: Predict the track duration (task3.ipynb)
