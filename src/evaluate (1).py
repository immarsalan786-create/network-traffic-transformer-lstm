
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
model = tf.keras.models.load_model('trained_model.h5')

X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
y_pred = np.argmax(model.predict(X_test), axis=1)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred, average='weighted'))
print("F1-Score :", f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))
