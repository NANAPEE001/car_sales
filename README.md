# Car Sales Prediction and Classification
This project involves analyzing and predicting car sales data using machine learning. The dataset includes various features such as the car model, company, price, engine type, transmission type, color, and more. The goal is to predict the car price using regression models and classify the car company using classification models. 

## Data Preprocessing
The dataset is first loaded and cleaned to handle missing and duplicate values. Various exploratory data analysis (EDA) techniques are applied to visualize the distribution of car attributes. The dataset is preprocessed by dropping unnecessary columns, encoding categorical variables using label encoding, and splitting the data into training and testing sets. Additionally, outliers are visualized, and feature engineering is done to make the data suitable for machine learning models.

## Regression Model for Price Prediction
Several regression models are tested to predict the price of the cars
Each model is evaluated using metrics like Mean Absolute Error (MAE) and R-squared (R2). The performance is visually represented using scatter plots comparing true vs predicted values.

## Classification Model for Car Company Prediction
A classification task was also performed to predict the car company based on other features. 
Model performance is assessed using accuracy scores and confusion matrices, and results are visualized using heatmaps.

## Key Insights
The project also includes visualizations to analyze the frequency of various car attributes (such as transmission type, engine type, and color) and their distribution based on price. The relationships between different variables, such as gender and body style, are explored with cross-tabulations and bar charts.
## Conclusion
By using various machine learning techniques, the project aims to predict car prices and classify car companies accurately. The best-performing models for both regression and classification tasks are selected based on the evaluation metrics. The insights derived from the EDA help in understanding the factors that influence car pricing and sales.
