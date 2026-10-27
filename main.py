import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. Загрузка данных MNIST (цифры 0–9)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Предобработка: нормализация и добавление канала
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28) -> (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)    # (10000, 28, 28) -> (10000, 28, 28, 1)

num_classes = 10

# 3. Построение простой CNN-модели
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 4. Обучение
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

# 5. Оценка на тестовой выборке
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# 6. Пример предсказания на нескольких изображениях
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)
true_labels = y_test[:5]

print("Predicted:", predicted_labels)
print("True:     ", true_labels)

# Для визуализации (опционально)
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.title(f"Pred: {predicted_labels[i]}\nTrue: {true_labels[i]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("predictions.png")
print("График сохранён как predictions.png")
