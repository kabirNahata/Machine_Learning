import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Load the CSV file
df = pd.read_csv('housing_data.csv')

# Step 2: Prepare the features and target
X = df[['Bedrooms', 'Washrooms']]
y = df['Price']

# Step 3: Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Visualize data
plt.scatter(df['Bedrooms'], df['Price'], color='blue', label='Bedrooms')
plt.scatter(df['Washrooms'], df['Price'], color='green', label='Washrooms')
plt.xlabel("Bedrooms / Washrooms")
plt.ylabel("Price")
plt.title("Housing Price Based on Features")
plt.legend()
plt.show()

# Step 5: Get user input
bedrooms = int(input("Enter number of bedrooms: "))
washrooms = int(input("Enter number of washrooms: "))

# Step 6: Predict the price without warning
input_data = pd.DataFrame([[bedrooms, washrooms]], columns=['Bedrooms', 'Washrooms'])
predicted_price = model.predict(input_data)[0]
print(f"Predicted House Price: ${predicted_price:,.2f}")
