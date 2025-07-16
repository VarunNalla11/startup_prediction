import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load data
data = pd.read_csv('/Users/varun/Desktop/varun_projects 2/proj3/50_Startups.csv')

# Extract features and target
x = data[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
y = data['Profit']

# Encode the "State" column using one-hot encoding
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), ['State'])],
    remainder='passthrough'
)
x_encoded = ct.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make prediction based on user inputs
rd_spend = float(input("Enter R&D Spend: "))
admin_spend = float(input("Enter Administration Spend: "))
marketing_spend = float(input("Enter Marketing Spend: "))
state = input("Enter State (California, Florida, New York): ")

# Encode the state input using the same OneHotEncoder
state_encoded = ct.named_transformers_['encoder'].transform(pd.DataFrame({'State': [state]}))


# Combine user inputs and transformed state input
user_input = np.hstack([state_encoded.toarray()[0], [rd_spend, admin_spend, marketing_spend]])


# Reshape the input for prediction
user_input = user_input.reshape(1, -1)

# Make prediction
prediction = model.predict(user_input)
print(f"Predicted Profit: {prediction[0]}")
