Key Scripts and Their Functions:

  `main.py`: The controller script. Running this executes the entire project pipeline from data loading to model evaluation.
  `eda.py`: Performs initial data exploration, creates visualizations, and generates insights into the factors driving employee turnover.
  `preprocessing.py`: Handles data cleaning (duplicate removal), encoding categorical variables (`salary`, `department`), and feature engineering (creating an `overworked` flag).
 `models.py`: Contains functions to train and tune multiple machine learning models (Logistic Regression, Decision Tree, Random Forest) using GridSearchCV..  `evaluation.py`: Evaluates model performance on test data, generates metrics (precision, recall, AUC), and creates confusion matrices and feature importance plots.
  `utils.py`: A helper module with utility functions for saving/loading models and compiling results to avoid code repetition.

 How to Run This Project

  Clone the repository
    ```bash
    git clone https://github.com/your-username/salifort-motors-employee-retention-ml.git
    cd salifort-motors-employee-retention-ml
    ```

  Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

  Run the main pipeline
    ```bash
    python src/main.py
    ```
    This will execute the entire analysis, from loading the data to generating evaluation plots.

 Key Insights & Results
-Primary Drivers of Turnover: Low satisfaction, excessive workload (>175 hrs/month), having 7 projects, and tenure between 4-6 years.
- Best Performing Model: Random Forest achieved the highest AUC score, effectively balancing precision and recall to identify at-risk employees.
- Business Impact: The model provides HR with a actionable tool to prioritize intervention efforts, potentially reducing turnover by targeting its root causes.

 Model Performance
The final Random Forest model was selected based on its strong performance on the test set:
High Recall: Effectively identifies most employees who are actually at risk of leaving (minimizes false negatives).
Strong AUC Score: Demonstrates excellent overall capability to distinguish between employees who will stay and those who will leave.

Recommendations for Salifort Motors
1.  Address Overwork: Implement policies to monitor and cap monthly hours, especially for employees with high project counts.
2.  Focus on Mid Tenure Employees: Develop retention packages and career development programs for employees with 4-6 years of tenure.
3.  Regular Satisfaction Monitoring: Use pulse surveys to identify dips in satisfaction early and intervene.
4.  Promote from Within: Create clear pathways for promotions to retain top talent who feel stagnant.

 Author
**Khawaja Abdul Raffay**


 License
This project is licensed under the MIT License - see the LICENSE file for details.

