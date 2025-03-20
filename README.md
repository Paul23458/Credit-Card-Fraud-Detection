# Credit-Card-Fraud-Detection
Detecting credit card fraud is crucial for maintaining the integrity of financial transactions and protecting consumers from unauthorized charges. This project aims to develop a robust machine learning model that can accurately identify fraudulent credit card transactions.

The project emphasizes:
*Accuracy in fraud detection.
*Feature engineering to process and analyze data effectively.
*Model interpretability, providing insights into the decision-making process.
*Generalization, ensuring the model performs well on unseen data.

# Technologies Used
List the programming languages, libraries, and tools used in the project, for example:
Programming Language: Python
Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost
Tools: Jupyter Notebook, IDE (if applicable)

# Classification Models used: 
Logistic Regression, Random Forest, Decision trees, KNeighbours, XGB 

# Dataset:https://www.kaggle.com/datasets/kartik2112/fraud-detection

## Steps to Run the Project
1.Install the required dependencies:
pip install -r requirements.txt
2. Run the data preprocessing script:
python preprocess_data.py
3. Train the model:
python train_model.py
4. Evaluate the model:
python evaluate_model.py
5. (Optional) Run the prediction script:
python predict.py


## Results
index,        Model,          Accuracy_Score
0,     LogisticRegression,       0.9941
1,    RandomForestClassifier,    0.9941
2,     DecisionTreeClassifier,   0.996
3,      KNeighborsClassifier,    0.9959
4,           XGBClassifier,      0.994

# Future Work
Mention potential improvements or next steps for the project:
Testing on larger, real-world datasets.
Deploying the model using Flask/Django for real-time fraud detection.
Enhancing explainability using tools like SHAP or LIME.
