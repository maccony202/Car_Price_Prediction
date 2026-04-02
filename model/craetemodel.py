import pickle as pk
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


def create_model(car_data):

    car_data = car_data[car_data['Price'] < 200000]

    x = car_data.drop(['Price', 'Model'], axis=1)
    y = np.log1p(car_data['Price'])

    x = pd.get_dummies(x)

    pk.dump(x.columns.tolist(), open("model_columns.pkl", "wb"))

    

    #data spliting
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    #scalling the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    #model training
    model = LinearRegression()
    model.fit(x_train, y_train)

    #saving the train model
    with open("model.pkl", "wb") as f:
        pk.dump(model,f)


    with open("scaler.pkl", "wb") as f:
        pk.dump(scaler, f)

    #Evaluating the mode
    y_predict = model.predict(x_test)

    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_predict)

    #testing for accuracy and classification
    print("MAE:", mean_absolute_error(y_test_actual, y_pred_actual))
    print("MSE:", mean_squared_error(y_test_actual, y_pred_actual))
    print("R2 Score:", r2_score(y_test_actual, y_pred_actual))

    return  model, scaler