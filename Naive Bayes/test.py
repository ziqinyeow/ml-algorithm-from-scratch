import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
# import tensorflow as tf

from NB import NaiveBayes


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


X, y = datasets.make_classification(
    n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print(accuracy(y_test, predictions))  # 0.81

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation="relu", input_shape=(10,)),
#     tf.keras.layers.Dense(32, activation="relu"),
#     tf.keras.layers.Dense(16, activation="relu"),
#     tf.keras.layers.Dense(8, activation="relu"),
#     tf.keras.layers.Dense(1, activation="sigmoid"),
# ])

# model.compile(
#     loss=tf.keras.losses.BinaryCrossentropy(),
#     optimizer=tf.keras.optimizers.Adam(),
#     metrics=["accuracy"]
# )


# model.fit(X_train, y_train, epochs=20)

# y_pred = tf.squeeze(model.predict(X_test))
# y_pred = tf.map_fn(lambda x: 1 if x >= 0.5 else 0, y_pred)

# print(accuracy(y_test, y_pred)) # 0.835
