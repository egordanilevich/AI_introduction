from math import sqrt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures




    


def learning(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=15)
    # print(X_train.shape, X_test.shape)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred) * 100
    r2_test = r2_score(y_test, y_test_pred) * 100
    rsd_train = (sqrt(mse_train) / np.mean(y_train)) * 100
    rsd_test = (sqrt(mse_test) / np.mean(y_test)) * 100

    metrics_dict = {'Средняя квадратическая ошибка (MSE)': [f"{mse_train:.3f}", f"{mse_test:.3f}"],
                'Средняя относительная погрешность (RSD)': [f"{rsd_train:.3f}%", f"{rsd_test:.3f}%"],
                'Коэффициент детерминации (R^2)': [f"{r2_train:.3f}%", f"{r2_test:.3f}%"]}
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index',
                                    columns=['обучающая', 'тестовая'])
    print(metrics_df)
    


if __name__ == "__main__":    
    data = pd.read_csv('insurance.csv')
    data.shape
    data.head()
    data['log_charges'] = np.log1p(data['charges']) # логарифмирование
    # гистограмма распределения    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    axes[0].hist(data['charges'], bins=15, color='blue', edgecolor='black')
    axes[0].set_xlabel('Charges')
    axes[0].set_ylabel('Frequency')
    axes[1].hist(data['log_charges'], bins=15, color='red', edgecolor='black')
    axes[1].set_xlabel('Log Charges')
    axes[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    data = data.drop('charges', axis=1)

    quantitative = ['age', 'bmi', 'children']  # количественные
    qualitative = ['sex', 'smoker', 'region']  # качетвенные

    scaled = MinMaxScaler().fit_transform(data[quantitative])# обезразмеривание
    scaled_df = pd.DataFrame(scaled, columns=quantitative)
    print("Только количественные признаки")
    learning(scaled_df, data['log_charges'])
    print()
    
    data_encoded = pd.get_dummies(data[qualitative], columns=qualitative, drop_first=True)
    data_df = pd.concat([data_encoded, scaled_df], axis=1)
    print("От всех признаков, включая качественные")
    learning(data_df, data['log_charges'])

    print()
    print("Полином")
    plf = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = plf.fit_transform(scaled_df)
    plf.get_feature_names_out(quantitative)
    X_poly_df = pd.DataFrame(X_poly, columns=plf.get_feature_names_out(quantitative))
    X_poly_encoded = pd.concat([X_poly_df, data_df], axis=1)
    
    learning(X_poly_encoded, data['log_charges'])

# R^2 представляет собой долю дисперсии зависимой переменной, которая может быть объяснена предсказанными значениями модели
# RSD позволяет измерить изменчивость или разброс данных относительно их среднего значения
