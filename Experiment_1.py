
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

np.random.seed(42)
tf.random.set_seed(42)

chars = "abcdefghijklmnopqrstuvwxyz"

START = 1
END = 2

char_to_int = {c: i + 3 for i, c in enumerate(chars)}
int_to_char = {i: c for c, i in char_to_int.items()}

max_len = 10
vocab_size = len(chars) + 3

words = []

for _ in range(20000):
    length = np.random.randint(3, max_len + 1)
    word = "".join(np.random.choice(list(chars), length))
    words.append(word)

encoder_input = np.zeros((len(words), max_len), dtype="int32")
decoder_input = np.zeros((len(words), max_len + 1), dtype="int32")
decoder_output = np.zeros((len(words), max_len + 1), dtype="int32")

for i, word in enumerate(words):
    reversed_word = word[::-1]

    encoder_input[i, :len(word)] = [char_to_int[c] for c in word]

    decoder_input[i, 0] = START
    decoder_input[i, 1:len(word) + 1] = [char_to_int[c] for c in reversed_word]

    decoder_output[i, :len(word)] = [char_to_int[c] for c in reversed_word]
    decoder_output[i, len(word)] = END

sample_weights = (decoder_output != 0).astype("float32")

encoder_inputs = Input(shape=(max_len,))
encoder_embedding = Embedding(vocab_size, 64, mask_zero=True)(encoder_inputs)

encoder_lstm = LSTM(128, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)

decoder_inputs = Input(shape=(max_len + 1,))
decoder_embedding = Embedding(vocab_size, 64, mask_zero=True)(decoder_inputs)

decoder_lstm = LSTM(128, return_sequences=True)
decoder_outputs = decoder_lstm(
    decoder_embedding,
    initial_state=[state_h, state_c]
)

decoder_outputs = Dense(vocab_size, activation="softmax")(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    [encoder_input, decoder_input],
    np.expand_dims(decoder_output, -1),
    sample_weight=sample_weights,
    epochs=25,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

def reverse_word(word):
    word = word.lower()

    if not word or len(word) > max_len or any(c not in char_to_int for c in word):
        return "Enter a word containing 1 to 10 English letters."

    input_seq = np.zeros((1, max_len), dtype="int32")
    input_seq[0, :len(word)] = [char_to_int[c] for c in word]

    decoder_seq = np.zeros((1, max_len + 1), dtype="int32")
    decoder_seq[0, 0] = START

    result = []

    for t in range(max_len + 1):
        prediction = model.predict(
            [input_seq, decoder_seq],
            verbose=0
        )

        next_char = np.argmax(prediction[0, t])

        if next_char == END or next_char == 0:
            break

        if next_char not in int_to_char:
            break

        result.append(int_to_char[next_char])

        if t + 1 < max_len + 1:
            decoder_seq[0, t + 1] = next_char

    return "".join(result)

test_words = ["hello", "world", "python", "machine", "learning"]

print("\nSequence-to-Sequence Word Reversal\n")

for word in test_words:
    print("Input:", word)
    print("Reversed Output:", reverse_word(word))
    print()

print("Experiment completed successfully!")
