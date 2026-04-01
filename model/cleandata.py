import pandas as pd


def  get_clean_data():
    car_data = pd.read_csv("../data/car_price_prediction.csv")
    car_data = car_data.drop(['ID'], axis=1)

    car_data = car_data.replace('-', None)
    # car_data['Price','Levy', 'Model', 'Mileage'] = pd.to_numeric(car_data['Price','Levy', 'Model', 'Mileage'], errors='coerce')
    car_data['Price'] = pd.to_numeric(car_data['Price'], errors='coerce')
    car_data['Levy'] = pd.to_numeric(car_data['Levy'], errors='coerce')

    car_data['Mileage'] = car_data['Mileage'].astype(str).str.replace(' km', '')
    car_data['Mileage'] = pd.to_numeric(car_data['Mileage'], errors='coerce')

    car_data['Leather interior'] = car_data['Leather interior'].map({'Yes': 1, 'No': 0})

    car_data = car_data.dropna(subset=['Price'])

    car_data['Levy'] = car_data['Levy'].fillna(car_data['Levy'].median())
    car_data['Mileage'] = car_data['Mileage'].fillna(car_data['Mileage'].median())

    return car_data