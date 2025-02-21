import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_xor_dataset():
    data = np.array([[int(bit) for bit in f"{num:04b}"] for num in range(16)])
    labels = np.array([[np.count_nonzero(sample) % 2] for sample in data])
    return data, labels

inputs, outputs = create_xor_dataset()

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=4))
model.add(Dropout(0.2))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(inputs, outputs, epochs=2000, batch_size=4, verbose=1)

predicted_values = model.predict(inputs)
predicted_labels = (predicted_values > 0.5).astype(int)

print("\nРезультати роботи нейронної мережі:")
for inp, pred in zip(inputs, predicted_labels):
    print(f"Вхідні дані: {inp} -> Передбачене значення: {pred[0]}")
