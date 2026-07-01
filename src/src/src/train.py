import numpy as np
from model import build_model
import pickle

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
num_classes = len(np.unique(y_train))

model = build_model((X_train.shape[1], X_train.shape[2]), num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)
model.save('trained_model.h5')

with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
