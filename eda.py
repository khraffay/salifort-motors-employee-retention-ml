# eda.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import save_plot  # Import the save function

def perform_eda(df0):
    print("Gathering basic information about the data")
    print(df0.info())
    
    print("\nGathering descriptive statistics about the data")
    print(df0.describe())
    
    print("\nDisplaying all column names")
    print(df0.columns)
    
    # Rename columns as needed
    df0 = df0.rename(columns={'Work_accident': 'work_accident',
                              'average_montly_hours': 'average_monthly_hours',
                              'time_spend_company': 'tenure',
                              'Department': 'department'})
    print("\nDisplaying all column names after renaming")
    print(df0.columns)
    
    print("\nChecking for missing values")
    print(df0.isna().sum())
    
    print("\nChecking for duplicates")
    print(df0.duplicated().sum())
    
    print("\nInspecting some rows containing duplicates")
    print(df0[df0.duplicated()].head())
    
    # Drop duplicates and save resulting dataframe in a new variable as needed
    df1 = df0.drop_duplicates(keep='first')
    print("\nFirst few rows after dropping duplicates")
    print(df1.head())
    
    # Create a boxplot to visualize distribution of `tenure` and detect any outliers
    plt.figure(figsize=(6,6))
    plt.title('Boxplot to detect outliers for tenure', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.boxplot(x=df1['tenure'])
    save_plot('tenure_boxplot.png')
    plt.show()
    
    # Determine the number of rows containing outliers 
    percentile25 = df1['tenure'].quantile(0.25)
    percentile75 = df1['tenure'].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    print("Lower limit:", lower_limit)
    print("Upper limit:", upper_limit)
    outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]
    print("Number of rows in the data containing outliers in `tenure`:", len(outliers))
    
    # Get numbers and percentages of people who left vs. stayed
    print("\nCounts of people who left vs. stayed:")
    print(df1['left'].value_counts())
    print("\nPercentages of people who left vs. stayed:")
    print(df1['left'].value_counts(normalize=True))
    
    # Create a plot as needed 
    fig, ax = plt.subplots(1, 2, figsize = (22,8))
    sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient="h", ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_title('Monthly hours by number of projects', fontsize='14')
    tenure_stay = df1[df1['left']==0]['number_project']
    tenure_left = df1[df1['left']==1]['number_project']
    sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[1])
    ax[1].set_title('Number of projects histogram', fontsize='14')
    save_plot('hours_vs_projects.png')
    plt.show()
    
    # Get value counts of stayed/left for employees with 7 projects
    print("\nValue counts for employees with 7 projects:")
    print(df1[df1['number_project']==7]['left'].value_counts())
    
    # Create scatterplot of `average_monthly_hours` versus `satisfaction_level`
    plt.figure(figsize=(16, 9))
    sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
    plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
    plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
    plt.title('Monthly hours by satisfaction level', fontsize='14')
    save_plot('hours_vs_satisfaction.png')
    plt.show()
    
    # Create a plot as needed 
    fig, ax = plt.subplots(1, 2, figsize = (22,8))
    sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left', orient="h", ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_title('Satisfaction by tenure', fontsize='14')
    tenure_stay = df1[df1['left']==0]['tenure']
    tenure_left = df1[df1['left']==1]['tenure']
    sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5, ax=ax[1])
    ax[1].set_title('Tenure histogram', fontsize='14')
    save_plot('satisfaction_vs_tenure.png')
    plt.show()
    
    # Calculate mean and median satisfaction scores
    print("\nMean and median satisfaction scores by left status:")
    print(df1.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median]))
    
    # Create a plot as needed 
    fig, ax = plt.subplots(1, 2, figsize = (22,8))
    tenure_short = df1[df1['tenure'] < 7]
    tenure_long = df1[df1['tenure'] > 6]
    sns.histplot(data=tenure_short, x='tenure', hue='salary', discrete=1, 
                 hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.5, ax=ax[0])
    ax[0].set_title('Salary histogram by tenure: short-tenured people', fontsize='14')
    sns.histplot(data=tenure_long, x='tenure', hue='salary', discrete=1, 
                 hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.4, ax=ax[1])
    ax[1].set_title('Salary histogram by tenure: long-tenured people', fontsize='14')
    save_plot('salary_by_tenure.png')
    plt.show()
    
    # Create scatterplot of `average_monthly_hours` versus `last_evaluation`
    plt.figure(figsize=(16, 9))
    sns.scatterplot(data=df1, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.4)
    plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
    plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
    plt.title('Monthly hours by last evaluation score', fontsize='14')
    save_plot('hours_vs_evaluation.png')
    plt.show()
    
    # Create plot to examine relationship between `average_monthly_hours` and `promotion_last_5years`
    plt.figure(figsize=(16, 3))
    sns.scatterplot(data=df1, x='average_monthly_hours', y='promotion_last_5years', hue='left', alpha=0.4)
    plt.axvline(x=166.67, color='#ff6361', ls='--')
    plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
    plt.title('Monthly hours by promotion last 5 years', fontsize='14')
    save_plot('hours_vs_promotion.png')
    plt.show()
    
    # Display counts for each department
    print("\nDepartment value counts:")
    print(df1["department"].value_counts())
    
    # Create stacked histogram to compare department distribution
    plt.figure(figsize=(11,8))
    sns.histplot(data=df1, x='department', hue='left', discrete=1, 
                 hue_order=[0, 1], multiple='dodge', shrink=.5)
    plt.xticks(rotation=45)
    plt.title('Counts of stayed/left by department', fontsize=14)
    save_plot('department_distribution.png')
    plt.show()
    
    # Plot a correlation heatmap (using only numeric columns)
    plt.figure(figsize=(16, 9))
    numeric_df = df0.select_dtypes(include=[np.number])
    heatmap = sns.heatmap(numeric_df.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12)
    save_plot('correlation_heatmap.png')
    plt.show()
    
    return df1