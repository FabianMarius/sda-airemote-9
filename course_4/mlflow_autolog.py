import mlflow
import mlflow.tensorflow
import tensorflow as tf

from pathlib import Path
# setez output-ul pt mllflow in directorul in care se aflta acest script (calea este calculata dinamic)
mlflow.set_tracking_uri(Path(__file__).parent / "mlruns")

# Enable auto logging
mlflow.set_experiment("Experiment - Autolog")
mlflow.tensorflow.autolog()

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
# Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

# Train the model
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5
)