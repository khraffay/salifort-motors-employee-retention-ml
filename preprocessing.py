# preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_plot  # Import the save function

def preprocess_data(df1):
    print("Starting data preprocessing...")
    
    # Copy the dataframe
    df_enc = df1.copy()

    # Encode the `salary` column as an ordinal numeric category
    df_enc['salary'] = (
        df_enc['salary'].astype('category')
        .cat.set_categories(['low', 'medium', 'high'])
        .cat.codes
    )

    # Dummy encode the `department` column
    df_enc = pd.get_dummies(df_enc, drop_first=False)

    # Display the new dataframe
    print("First few rows of encoded dataframe:")
    print(df_enc.head())
    
    # Create a heatmap to visualize how correlated variables are
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_enc[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure']]
                .corr(), annot=True, cmap="crest")
    plt.title('Heatmap of the dataset')
    save_plot('numeric_correlation_heatmap.png')
    plt.show()
    
    # Create a stacked bar plot to visualize number of employees across department
    pd.crosstab(df1['department'], df1['left']).plot(kind ='bar',color='mr')
    plt.title('Counts of employees who left versus stayed across department')
    plt.ylabel('Employee count')
    plt.xlabel('Department')
    save_plot('department_crosstab.png')
    plt.show()
    
    # Select rows without outliers in `tenure` and save resulting dataframe in a new variable
    percentile25 = df1['tenure'].quantile(0.25)
    percentile75 = df1['tenure'].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    
    df_logreg = df_enc[(df_enc['tenure'] >= lower_limit) & (df_enc['tenure'] <= upper_limit)]
    print("First few rows of dataframe for logistic regression (without tenure outliers):")
    print(df_logreg.head())
    
    return df_enc, df_logreg, lower_limit, upper_limit


def engineer_features(df_enc):
    print("Starting feature engineering...")
    
    # Drop `satisfaction_level` and save resulting dataframe in new variable
    df2 = df_enc.drop('satisfaction_level', axis=1)
    print("First few rows after dropping satisfaction_level:")
    print(df2.head())
    
    # Create `overworked` column. For now, it's identical to average monthly hours.
    df2['overworked'] = df2['average_monthly_hours']
    print('Max hours:', df2['overworked'].max())
    print('Min hours:', df2['overworked'].min())
    
    # Define `overworked` as working > 175 hrs/week
    df2['overworked'] = (df2['overworked'] > 175).astype(int)
    print("First few rows of new 'overworked' column:")
    print(df2['overworked'].head())
    
    # Drop the `average_monthly_hours` column
    df2 = df2.drop('average_monthly_hours', axis=1)
    print("First few rows of final engineered dataframe:")
    print(df2.head())
    
    return df2