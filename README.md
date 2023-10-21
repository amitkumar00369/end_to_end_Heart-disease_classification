# end_to_end_Heart-disease_classification
# End-to-End Heart Disease Classification

This project is an end-to-end heart disease classification using six different machine learning algorithms: Decision Tree, K-Nearest Neighbors (KNN), Random Forest, Support Vector Classification (SVC), Linear Support Vector Classification (LinearSVC), and Logistic Regression. The goal of this project is to build models that can accurately predict the presence or absence of heart disease based on a set of features, achieving an impressive accuracy score of 88.86% by Logistic Regression.

## Dataset

The dataset used for this project is the [Heart Disease UCI dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) from the UCI Machine Learning Repository. It contains various attributes related to heart health, and the target variable indicates the presence or absence of heart disease.

## Model Performance

After extensive data preprocessing, model training, and evaluation, the six machine learning models achieved an accuracy score of 88.86% on the heart disease classification task. Here are the model-specific scores:

1. **Decision Tree**: Achieved an accuracy score of X%.

2. **K-Nearest Neighbors (KNN)**: Achieved an accuracy score of X%.

3. **Random Forest**: Achieved an accuracy score of X%.

4. **Support Vector Classification (SVC)**: Achieved an accuracy score of X%.

5. **Linear Support Vector Classification (LinearSVC)**: Achieved an accuracy score of X%.

6. **Logistic Regression**: Achieved an accuracy score of X%.

The models were evaluated not only for accuracy but also for precision, recall, and F1-score. The final model was selected based on these comprehensive performance metrics, and it achieved an overall accuracy of 88.86%.

## Dependencies

Before running the code, ensure that you have the following Python libraries installed:

- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn

You can install these dependencies using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Project Structure

The project directory is structured as follows:

```
|-- heart_disease_classification/
    |-- data/
        |-- heart.csv
    |-- src/
        |-- heart_disease_classification.py
    |-- README.md
```

- `data/`: This directory contains the dataset file (`heart.csv`).
- `src/`: This directory contains the Python script (`heart_disease_classification.py`) that performs the data preprocessing, model building, and evaluation.

## Running the Code

To reproduce the model performance and accuracy score of 88.86%, follow these steps:

1. Clone or download this repository to your local machine.

2. Navigate to the `src/` directory.

3. Run the Python script:

```bash
python heart_disease_classification.py
```

The script will load the dataset, preprocess the data, split it into training and testing sets, build the models, and evaluate their performance.

## Conclusion

This project demonstrates the successful implementation of various machine learning algorithms for heart disease classification. With an overall accuracy score of 88.86%, it provides a robust solution for predicting heart disease, helping healthcare professionals make informed decisions and potentially save lives.

Feel free to explore the code and the dataset to adapt this solution for your own classification tasks or to further enhance the project.
