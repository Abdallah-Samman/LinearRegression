# LinearRegression
Appling Linear Regression on Airbnb dataset to predict which neighborhood have the highest future price potential.

# Airbnb Price Prediction Project

This project focuses on predicting Airbnb rental prices using a linear regression model. The goal is to analyze various features related to listings and determine their influence on pricing.

## Project Overview

1. Import necessary libraries and modules:
    - `pandas` for data manipulation and analysis
    - `matplotlib.pyplot` and `seaborn` for data visualization
    - `sklearn` for implementing the linear regression model

2. Load the dataset and select relevant columns for analysis:
    - The dataset is assumed to be loaded into a `dataset` variable.
    - The `selected_columns` list contains the column names required for analysis.

3. Create a new dataframe with only the selected columns.

4. Handle missing values:
    - Drop rows with missing values using `dropna()`.

5. Convert categorical variables to numerical using one-hot encoding:
    - Use `pd.get_dummies()` to convert the 'neighbourhood' column.

6. Separate the target variable ('price') from the features:
    - Assign the features to `X` and the target variable to `y`.

7. Split the data into training and testing sets:
    - Use `train_test_split()` to split the data into 60% for training and 40% for testing.

8. Create a linear regression model:
    - Instantiate a `LinearRegression` object.

9. Perform feature selection using a threshold:
    - Use `SelectFromModel` with the linear regression model and a threshold set to 'median' to select features.
    - Fit the selector on the training data and transform the test data.

10. Fit the linear regression model with the selected features:
    - Use `fit()` on the linear regression model with the selected features and the target variable.

11. Get the coefficients of the linear regression model:
    - Access the coefficients using the `coef_` attribute.

12. Get the original feature names from the selector:
    - Retrieve the selected feature names using `selector.get_support()`.

13. Create a dataframe to display the feature coefficients:
    - Sort the dataframe by the coefficients in descending order.

14. Plot the neighborhood coefficients:
    - Filter the dataframe to include only neighborhood features.
    - Create a bar plot using `sns.barplot()` with customized styling.
    - Invert the y-axis and adjust the labels and title.

## Results

The project aims to determine the influence of neighborhood features on Airbnb rental prices. By analyzing the coefficients obtained from the linear regression model, we can identify the neighborhoods that contribute the most to the predicted prices. The bar plot visualizes the coefficients, with darker colors indicating higher coefficients.

![Neighborhood Coefficients](path/to/neighborhood_coefficients_plot.png)

In the plot, each bar represents a neighborhood feature, and the height of the bar represents the coefficient. Positive coefficients indicate that the neighborhood positively influences the price, while negative coefficients indicate a negative influence. The bar plot allows us to identify the neighborhoods that have the most significant impact on Airbnb rental prices.

This project provides valuable insights for Airbnb hosts and stakeholders who can use the analysis to optimize pricing strategies and understand the factors affecting rental prices.

For the complete code and more details, please refer to the [Jupyter Notebook](path/to/your/notebook.ipynb).

For any questions or further information, feel free to reach out to me. Happy analyzing!

