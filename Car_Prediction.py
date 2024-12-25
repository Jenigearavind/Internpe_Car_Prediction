from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
car = pd.read_csv('/content/quikr_car.csv')
backup = car.copy()
car['year'] = car['year'].replace('...', np.nan)
car['year'] = pd.to_numeric(car['year'], errors='coerce')
car = car.dropna(subset=['year'])
car['year'] = car['year'].astype(int)
car = car[car['Price'] != 'Ask For Price']
car['Price'] = car['Price'].str.replace(',', '').astype(int)
car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',', '')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
car = car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
car = car.reset_index(drop=True)
car = car[car['Price'] < 6000000]
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
numeric_features = ['year', 'kms_driven']
categorical_features = ['name', 'company', 'fuel_type']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
def user_input():
    print("Enter the details of the car for price prediction:")
    name = input("Car Name: ")
    company = input("Car Company: ")
    year = int(input("Year of Manufacture: "))
    kms_driven = int(input("Kilometers Driven: "))
    fuel_type = input("Fuel Type: ")    
    return pd.DataFrame({
        'name': [name],
        'company': [company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    })
user_car = user_input()
predicted_price = model.predict(user_car)
print(f"The predicted price for the car is: ₹{predicted_price[0]:,.2f}")
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 7))
ax = sns.boxplot(x='company', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
plt.title('Car Price Distribution by Company')
plt.show()
plt.figure(figsize=(15, 7))
ax = sns.barplot(x='company', y='Price', data=car, estimator='mean')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
plt.title('Average Car Price by Company')
plt.show()
plt.figure(figsize=(15, 7))
ax = sns.violinplot(x='company', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
plt.title('Car Price Distribution by Company (Violin Plot)')
plt.show()
plt.figure(figsize=(15, 7))
ax = sns.scatterplot(x='company', y='Price', data=car)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
plt.title('Car Prices by Company (Scatter Plot)')
plt.show()

