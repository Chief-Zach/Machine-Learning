from keras.preprocessing import sequence
import tensorflow as tf
from keras.datasets import imdb
VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 128

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),  # Graph vector form, 32 dimensions
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Between 0-1
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])




history = model.fit(x=train_data, y=train_labels, batch_size=BATCH_SIZE, epochs=10, validation_split=0.2)

results = model.evaluate(test_data, test_labels)
print(results)

