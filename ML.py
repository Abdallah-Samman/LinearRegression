import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
import numpy as np

np.random.seed(1999166)

dataset = pd.read_csv('Airbnb.csv')

# Select relevant columns for analysis
selected_columns = ['neighbourhood', 'minimum_nights', 'maximum_nights', 'availability_30', 'availability_365',
                    'number_of_reviews', 'review_scores_rating', 'price']

# Create a new dataframe with only the selected columns
df = dataset[selected_columns]

# Handle missing values, if any
df = df.dropna()

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['neighbourhood'])

# Separate the target variable (price) from the features
X = df.drop('price', axis=1)
y = df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.60, random_state=1999166)

# Create a linear regression model
linear_model = LinearRegression()

# Perform feature selection using a threshold
selector = SelectFromModel(linear_model, threshold='median')
selector.fit(X_train, y_train)
X_selected = selector.transform(X_test)

# Fit the linear regression model with the selected features
linear_model.fit(X_selected, y_test)

# Get the coefficients of the linear regression model
coefficients = linear_model.coef_

# Get the original feature names from the selector
selected_feature_names = X.columns[selector.get_support()]

# Create a dataframe to display the feature coefficients
feature_coefficients_df = pd.DataFrame({'Feature': selected_feature_names, 'Coefficient': coefficients})
feature_coefficients_df = feature_coefficients_df.sort_values('Coefficient', ascending=False)

# Set the plot style and color palette
sns.set(style="whitegrid")
custom_palette = sns.color_palette("Reds_r", len(feature_coefficients_df))
custom_palette[0] = (0.6, 0.1, 0.1)  # Darken the color of the minimum value

# Filter for neighborhood features
neighborhood_coefficients_df = feature_coefficients_df[feature_coefficients_df['Feature'].str.startswith('neighbourhood_')]

# Create a figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the neighborhood coefficients as a bar plot with 15% transparency and inverted y-axis
sns.barplot(x='Coefficient', y='Feature', data=neighborhood_coefficients_df, palette=custom_palette, ax=ax)

# Modify the y-axis tick labels for a professional look
tick_labels = [feature.replace('neighbourhood_', '').replace('_', ' ').title() for feature in neighborhood_coefficients_df['Feature']]
ax.set_yticklabels(tick_labels)

# Invert the y-axis to have higher coefficients at the top
ax.invert_yaxis()

# Reverse the y-axis limits
ax.set_ylim(ax.get_ylim()[::-1])

# Format the axis labels and title
ax.set_xlabel('Coefficient')
ax.set_ylabel('Neighborhood')
ax.set_title('Neighborhood Coefficients for Predicted Price')

# Adjust spacing to avoid overlapping labels
plt.tight_layout()

# Show the plot
plt.show()
