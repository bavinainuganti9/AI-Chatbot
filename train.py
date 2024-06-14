import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

training = np.load('training_data.npy')
train_x = training[:, :len(training[0]) // 2]
train_y = training[:, len(training[0]) // 2:]

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1, validation_data=(val_x, val_y), callbacks=[early_stopping, model_checkpoint])

model.save('chatbot_model.h5')
print('Done')
