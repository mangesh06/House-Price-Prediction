# House-Price-Prediction Project
Predict house prices efficiently using advanced machine learning algorithms. This project delves into data preprocessing techniques and sophisticated models to ensure accurate and insightful predictions.

## Introduction
This project is designed for data science enthusiasts to tackle a regression and time series forecasting challenge, perfect for those with a background in Python and basic machine learning concepts. Aimed at students who have completed an online machine learning course, the project focuses on predicting house prices in Ames, Iowa, using 79 explanatory variables. Participants will refine their feature engineering skills and address missing data issues while experimenting with advanced regression techniques like random forest and gradient boosting.

## Objective
1. **Explore Various Regression Methods:** Employ different regression methods, including Random Forest Regressor, Decision Tree Regressor, MLP Regressor, Support Vector Regressor, and Gradient Descent Regressor. Enhance predictive accuracy through hyperparameter tuning with cross-validation.
2. **Ensemble Predictions via Stacking:** Implement one-layer stacking by combining outputs from top-performing algorithms such as Decision Tree, Random Forest, and Support Vector Machines. Experiment with various combinations to find the optimal predictive performance.
3. **Hyperparameter Tuning for Stacked Model:** Perform hyperparameter tuning with cross-validation specifically for the stacked model, refining parameters for optimal performance.

## Dataset
The project utilizes the Ames Housing dataset, created by Dean De Cock for educational purposes in data science. With 79 explanatory variables covering nearly all aspects of residential homes in Ames, Iowa, this dataset is a central element of a Kaggle competition. The task is to predict the final price of each home, encouraging creative feature engineering and the use of advanced regression techniques like random forest and gradient boosting. For more details and access to the dataset, visit the Kaggle competition page: [House Prices Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Python Packages Used
1. vecstack
2. pandas
3. numpy
4. plotly.express
5. LabelEncoder, OneHotEncoder from sklearn.preprocessing
6. train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score from sklearn.model_selection
7. RandomForestRegressor, GradientBoostingRegressor from sklearn.ensemble
8. DecisionTreeRegressor from sklearn.tree
9. MLPRegressor from sklearn.neural_network
10. mean_squared_error, accuracy_score from sklearn.metrics

## Result
The project resulted in refined predictive models utilizing advanced regression techniques. By applying creative feature engineering and experimenting with algorithms such as random forest and gradient boosting, the models achieved greater accuracy in predicting house prices. Evaluation metrics like mean squared error highlighted the models' effectiveness in capturing complex patterns in the Ames Housing dataset. Ensemble predictions through one-layer stacking integrated insights from decision tree, random forest, and support vector machines. Comprehensive hyperparameter tuning further optimized the stacked model, enhancing its predictive performance.