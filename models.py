# models.py
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import time
from utils import make_results

def prepare_data(df, target_col='left'):
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    return X_train, X_test, y_train, y_test

def train_logistic_regression(df_logreg, lower_limit, upper_limit):
    print("\n" + "="*50)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*50)
    
    # Isolate the outcome variable
    y = df_logreg['left']
    print("First few rows of the outcome variable:")
    print(y.head())
    
    # Select the features
    X = df_logreg.drop('left', axis=1)
    print("First few rows of the selected features:")
    print(X.head())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    
    # Construct a logistic regression model and fit it
    log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
    
    # Get predictions
    y_pred = log_clf.predict(X_test)
    
    # Compute values for confusion matrix
    log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)
    log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=log_clf.classes_)
    log_disp.plot(values_format='')
    plt.title("Logistic Regression Confusion Matrix")
    plt.show()
    
    # Check class balance
    print("Class balance in logistic regression data:")
    print(df_logreg['left'].value_counts(normalize=True))
    
    # Create classification report
    target_names = ['Predicted would not leave', 'Predicted would leave']
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return log_clf, X_test, y_test

def train_decision_tree(X_train, y_train, model_name_suffix=""):
    print(f"\n" + "="*50)
    print(f"TRAINING DECISION TREE MODEL {model_name_suffix}")
    print("="*50)
    
    # Instantiate model
    tree = DecisionTreeClassifier(random_state=0)
    
    # Assign hyperparameters to search over
    cv_params = {'max_depth':[4, 6, 8, None],
                 'min_samples_leaf': [2, 5, 1],
                 'min_samples_split': [2, 4, 6]
                 }
    
    # Assign scoring metrics to capture
    scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
    
    # Instantiate GridSearch
    tree_model = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')
    
    # Train model with timing
    start_time = time.time()
    tree_model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Decision Tree {model_name_suffix} training time: {end_time - start_time:.2f} seconds")
    
    # Check best parameters and score
    print(f"Best parameters for Decision Tree {model_name_suffix}:")
    print(tree_model.best_params_)
    print(f"Best AUC score on CV for Decision Tree {model_name_suffix}: {tree_model.best_score_}")
    
    # Get CV results
    tree_cv_results = make_results(f'decision tree{model_name_suffix} cv', tree_model, 'auc')
    
    return tree_model, tree_cv_results

def train_random_forest(X_train, y_train, model_name_suffix=""):
    print(f"\n" + "="*50)
    print(f"TRAINING RANDOM FOREST MODEL {model_name_suffix}")
    print("="*50)
    
    # Instantiate model
    rf = RandomForestClassifier(random_state=0)
    
    # Assign hyperparameters to search over
    cv_params = {'max_depth': [3,5, None], 
                 'max_features': [1.0],
                 'max_samples': [0.7, 1.0],
                 'min_samples_leaf': [1,2,3],
                 'min_samples_split': [2,3,4],
                 'n_estimators': [300, 500],
                 }  
    
    # Assign scoring metrics to capture
    scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
    
    # Instantiate GridSearch
    rf_model = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')
    
    # Train model with timing
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Random Forest {model_name_suffix} training time: {end_time - start_time:.2f} seconds")
    
    # Check best parameters and score
    print(f"Best parameters for Random Forest {model_name_suffix}:")
    print(rf_model.best_params_)
    print(f"Best AUC score on CV for Random Forest {model_name_suffix}: {rf_model.best_score_}")
    
    # Get CV results
    rf_cv_results = make_results(f'random forest{model_name_suffix} cv', rf_model, 'auc')
    
    return rf_model, rf_cv_results