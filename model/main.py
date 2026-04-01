from cleandata import get_clean_data
from craetemodel import create_model
import pickle as pk

def main():
    car_data = get_clean_data()
    print(car_data.head())

    create_model(car_data)

    




if __name__ == '__main__':
    main()