# evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import get_scores, write_pickle, read_pickle, make_results, save_plot

def evaluate_models(tree1, rf1, tree2, rf2, X_test, y_test, X_test2, y_test2):
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Define path for saving models (current directory)
    path = ''
    
    # Save and load Random Forest model 1
    write_pickle(path, rf1, 'hr_rf1')
    rf1 = read_pickle(path, 'hr_rf1')
    
    # Get all CV results
    print("CV Results for Model 1 (Original Features):")
    tree1_cv_results = make_results('decision tree1 cv', tree1, 'auc')
    rf1_cv_results = make_results('random forest1 cv', rf1, 'auc')
    print(tree1_cv_results)
    print(rf1_cv_results)
    
    # Get test results for Random Forest 1
    rf1_test_scores = get_scores('random forest1 test', rf1, X_test, y_test)
    print("Test Results for Random Forest 1:")
    print(rf1_test_scores)
    
    # Save and load Random Forest model 2
    write_pickle(path, rf2, 'hr_rf2')
    rf2 = read_pickle(path, 'hr_rf2')
    
    # Get all CV results for Model 2
    print("\nCV Results for Model 2 (Engineered Features):")
    tree2_cv_results = make_results('decision tree2 cv', tree2, 'auc')
    rf2_cv_results = make_results('random forest2 cv', rf2, 'auc')
    print(tree2_cv_results)
    print(rf2_cv_results)
    
    # Get test results for Random Forest 2
    rf2_test_scores = get_scores('random forest2 test', rf2, X_test2, y_test2)
    print("Test Results for Random Forest 2:")
    print(rf2_test_scores)
    
    # Generate confusion matrix for Random Forest 2
    preds = rf2.best_estimator_.predict(X_test2)
    cm = confusion_matrix(y_test2, preds, labels=rf2.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf2.classes_)
    disp.plot(values_format='')
    plt.title("Random Forest 2 Confusion Matrix")
    save_plot('rf2_confusion_matrix.png')
    plt.show()
    
    return tree1_cv_results, rf1_cv_results, tree2_cv_results, rf2_cv_results, rf1_test_scores, rf2_test_scores

def visualize_models(tree2, rf2, X, X2):
    print("\n" + "="*50)
    print("MODEL VISUALIZATION")
    print("="*50)
    
    # Plot the decision tree
    plt.figure(figsize=(85,20))
    plot_tree(tree2.best_estimator_, max_depth=6, fontsize=14, feature_names=X2.columns, 
              class_names={0:'stayed', 1:'left'}, filled=True)
    plt.title("Decision Tree Visualization (Max Depth = 6)")
    save_plot('decision_tree_visualization.png')
    plt.show()
    
    # Get feature importances for decision tree
    tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, 
                                     columns=['gini_importance'], 
                                     index=X2.columns)
    tree2_importances = tree2_importances.sort_values(by='gini_importance', ascending=False)
    tree2_importances = tree2_importances[tree2_importances['gini_importance'] != 0]
    
    print("Decision Tree Feature Importances:")
    print(tree2_importances)
    
    # Plot decision tree feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(data=tree2_importances, x="gini_importance", y=tree2_importances.index, orient='h')
    plt.title("Decision Tree: Feature Importances for Employee Leaving", fontsize=12)
    plt.ylabel("Feature")
    plt.xlabel("Importance")
    save_plot('decision_tree_feature_importance.png')
    plt.show()
    
    # Get feature importances for random forest
    feat_impt = rf2.best_estimator_.feature_importances_
    ind = np.argpartition(rf2.best_estimator_.feature_importances_, -10)[-10:]
    feat = X.columns[ind]
    feat_impt = feat_impt[ind]
    
    y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
    y_sort_df = y_df.sort_values("Importance")
    
    # Plot random forest feature importances
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    y_sort_df.plot(kind='barh', ax=ax1, x="Feature", y="Importance")
    ax1.set_title("Random Forest: Feature Importances for Employee Leaving", fontsize=12)
    ax1.set_ylabel("Feature")
    ax1.set_xlabel("Importance")
    save_plot('random_forest_feature_importance.png')
    plt.show()