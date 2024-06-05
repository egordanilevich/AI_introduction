"""
Created on

Лабораторная работа №4 "Классификация с помощью логистической регрессии"

@author: Egor
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve

def print_count(series):
    counts = series.value_counts()
    for i in counts.index:
        print(f"Class {i}: {counts.loc[i]}")

if __name__ == "__main__": 
    """
        1. Загрузить данные из файла heart.csv, используя функцию read_csv() библиотеки pandas. 
        Более подробная информация о датасете https://www.kaggle.com/fedesoriano/heart-failure-prediction
        
        2. Проверить сбалансированность выборки (Сколько объектов класса "1" и "0" соответственно).
    """
    data = pd.read_csv('heart.csv')
    class_counts = data['target'].value_counts()
    class_fractions = class_counts / len(data)
    print(f"Класс 0: {class_fractions[0]:.3f} \nКласс 1: {class_fractions[1]:.3f}\n")

    """
        3. Выделить качественные признаки, провести их кодирование. 
        Для этого можно воспользоваться фунцией get_dummies() библиотеки pandas
    """
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    encoded_data = pd.get_dummies(data, columns=categorical_features)
    print(encoded_data.head(), "\n")
    
    """
        4. Провести нормирование всех количественных признаков. 
        В данном случае, чтобы не изменились значения кодированных качественных признаков 
        можно использовать класс MinMaxScaler()
    """
    print("Нормализация: ")
    scaler = MinMaxScaler()
    quantitative_features = [col for col in encoded_data.columns if col not in categorical_features]
    encoded_data[quantitative_features] = scaler.fit_transform(encoded_data[quantitative_features])
    print(encoded_data.head(), "\n")

    """
        5. Разделить выборку на обучающую и тестовую, используя функцию train_test_split().
        Параметр рандомизации задать random_state = 13 для возможности сравнения результатов.
    """
    print("Разделение выборок:")
    x_train, x_test, y_train, y_test = train_test_split(encoded_data.drop('target', axis=1),
                                                        encoded_data['target'],
                                                        test_size=0.3, 
                                                        random_state = 13)
    print("Тренировочная:") 
    print_count(y_train) 
    print("Тестовая:")
    print_count(y_test)
    print()
    
    """
    6. Обучить классификатор на основе логистической регрессии. 
    Использовать класс по вариантам: 1 - LogisticRegression; 2 - SGDClassifier. 
    Процедуру регуляризации не проводить. 
    Рассчитать предсказанные значения классов для тестовой выборки.
    
    7. Рассчитать точность классификации,
    в качестве метрики использовать долю верных ответов модели с реальными заначениями.
    """
    print("Логическая регрессия:")
    logic_classifier = LogisticRegression(penalty=None)
    logic_classifier.fit(x_train, y_train)
    y_pred = logic_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность классификации: {accuracy:.3f}\n")
    
    print("СГС классификация:")
    SGD_classifier = SGDClassifier(loss='log_loss', penalty=None, random_state=13)
    SGD_classifier.fit(x_train, y_train)
    y_pred = SGD_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность классификации: {accuracy:.3f}\n")

    """
       8.  Для заданного в диапазоне  [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]
        параметра регуляризации провести процедуру валидации,
        выбрать оптимальное значение гиперпараметра.
        Обучить модель при этом значении и рассчитать точность. 
        Для обучения также использовать классы LogisticRegression или SGDClassifier
    """
    lamb = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]
    
    print("Логическая регрессия:")
    acc_2 = np.zeros(len(lamb))
    for i, C in enumerate(lamb):
        logic_classifier = LogisticRegression(C=C)
        logic_classifier.fit(x_train, y_train)
        y_pred = logic_classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        acc_2[i] = accuracy
        print(f"Точность классификации при C={C}:    {accuracy:.3f}")

    logic_best_lamb = lamb[np.argmax(acc_2)]
    logic_best_accuracy = acc_2[np.argmax(acc_2)]
    print(f"\nОптимальное значение гиперпараметра: C={logic_best_lamb}, точность классификации: {logic_best_accuracy:.3f}\n")
    
    print("СГС классификация:")
    acc_2 = np.zeros(len(lamb))
    for i, alpha in enumerate(lamb):
        SGD_classifier = SGDClassifier(alpha=alpha, random_state=13)
        SGD_classifier.fit(x_train, y_train)
        y_pred = SGD_classifier.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        acc_2[i] = accuracy
        print(f"Точность классификации при alpha={alpha}:    {accuracy:.3f}")
    SGD_best_lamb_index = np.argmax(acc_2)
    SGD_best_lamb = lamb[SGD_best_lamb_index]
    SGD_best_accuracy = acc_2[SGD_best_lamb_index]
    print(f"\nОптимальное значение гиперпараметра: C={SGD_best_lamb}, точность классификации: {SGD_best_accuracy:.3f}")
    
    
    """
        9. Для лучшей модели изобразить матрицу ошибок (confusion matrix)
    """
    logic_best_classifier = LogisticRegression(C=logic_best_lamb)
    logic_best_classifier.fit(x_train, y_train)
    y_pred = logic_best_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{accuracy:.3f}")
   
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=logic_best_classifier.classes_)
    disp.plot(cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    
    SGD_best_classifier = SGDClassifier(alpha = SGD_best_lamb, random_state=13)
    SGD_best_classifier.fit(x_train, y_train)
    y_pred = SGD_best_classifier.predict(x_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = SGD_best_classifier.classes_)
    disp.plot(cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Best SGD class Confusion Matrix')
    plt.show()
    """
        10. Обучить модель, используя процедуру кросс-валидации с помощью класса LogisticRegressionCV. 
        Оценить точность модели.
    """
    classifier_cv = LogisticRegressionCV(Cs=lamb, cv=5, random_state=13)
    classifier_cv.fit(x_train, y_train)
    y_pred_cv = classifier_cv.predict(x_test)
    accuracy_cv = accuracy_score(y_test, y_pred_cv)
    print(f"Точность LogisticRegressionCV: {accuracy_cv:.3f} параметр {classifier_cv.C_}")
    
    """
        11. Для оценки точности построить ROC-кривую. 
        Для этого рассчитать предсказанные вероятности по последней модели.
        Рассчитать значения оценок FPR и TPR с помощью функции roc_curve().
    """
    y_prob = classifier_cv.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

