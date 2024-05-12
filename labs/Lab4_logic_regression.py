import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve

data = pd.read_csv('heart.csv')
class_counts = data['target'].value_counts()
class_fractions = class_counts / len(data)
print(f"Class 0: {class_fractions[0]:.3f}\nClass 1: {class_fractions[1]:.3f}\n")

categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
encoded_data = pd.get_dummies(data, columns=categorical_features)
# print(encoded_data.head())

scaler = MinMaxScaler()
quantitative_features = [col for col in encoded_data.columns if col not in categorical_features]
encoded_data[quantitative_features] = scaler.fit_transform(encoded_data[quantitative_features])
# print(encoded_data.head())

X_train, X_test, y_train, y_test = train_test_split(encoded_data.drop('target', axis=1),
                                                    encoded_data['target'], test_size=0.3)
# print(X_train.shape, X_test.shape, "\n")

classifier = LogisticRegression(penalty=None)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность классификации: {accuracy:.3f}")

lamb = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]
acc_2 = np.zeros(len(lamb))
for i, C in enumerate(lamb):
    classifier = LogisticRegression(C=C)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    acc_2[i] = accuracy
    print(f"Точность классификации при C={C}:    {accuracy:.3f}")

best_lamb = lamb[np.argmax(acc_2)]
best_accuracy = acc_2[np.argmax(acc_2)]
print(f"\nОптимальное значение гиперпараметра: C={best_lamb}, точность классификации: {best_accuracy:.3f}")

best_classifier = LogisticRegression(C=best_lamb)
best_classifier.fit(X_train, y_train)

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=best_classifier.classes_)
disp.plot(cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# plt.show()

classifier_cv = LogisticRegressionCV(Cs=lamb, cv=5)
classifier_cv.fit(X_train, y_train)
y_pred_cv = classifier_cv.predict(X_test)
accuracy_cv = accuracy_score(y_test, y_pred_cv)
print(f"Точность LogisticRegressionCV: {accuracy_cv:.3f} параметр {classifier_cv.C_}")

y_prob = classifier_cv.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
# plt.show()
