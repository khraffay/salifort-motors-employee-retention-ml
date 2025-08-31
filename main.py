# main.py
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

# Import custom modules
from eda import perform_eda
from preprocessing import preprocess_data, engineer_features
from models import prepare_data, train_logistic_regression, train_decision_tree, train_random_forest
from evaluation import evaluate_models, visualize_models
from utils import make_results

def main():
    print("SALIFORT MOTORS EMPLOYEE TURNOVER PREDICTION PROJECT")
    print("="*60)
    
    # Load dataset
    print("Loading dataset...")
    df0 = pd.read_csv("HR_capstone_dataset.csv")
    print("First few rows of the dataset:")
    print(df0.head())
    
    # Perform EDA
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    df1 = perform_eda(df0)
    
    # Preprocess data
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    df_enc, df_logreg, lower_limit, upper_limit = preprocess_data(df1)
    
    # Prepare data for machine learning models
    X_train, X_test, y_train, y_test = prepare_data(df_enc)
    
    # Train logistic regression
    log_clf, X_test_lr, y_test_lr = train_logistic_regression(df_logreg, lower_limit, upper_limit)
    
    # Train first set of models (original features)
    tree1, tree1_cv_results = train_decision_tree(X_train, y_train, "1")
    rf1, rf1_cv_results = train_random_forest(X_train, y_train, "1")
    
    # Engineer features and prepare new dataset
    df2 = engineer_features(df_enc)
    X_train2, X_test2, y_train2, y_test2 = prepare_data(df2)
    
    # Train second set of models (engineered features)
    tree2, tree2_cv_results = train_decision_tree(X_train2, y_train2, "2")
    rf2, rf2_cv_results = train_random_forest(X_train2, y_train2, "2")
    
    # Evaluate all models
    results = evaluate_models(tree1, rf1, tree2, rf2, X_test, y_test, X_test2, y_test2)
    
    # Visualize models
    visualize_models(tree2, rf2, X_train, X_train2)
    
    print("\n" + "="*50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()