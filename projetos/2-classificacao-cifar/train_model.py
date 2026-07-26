import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
import io
import numpy as np

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_and_split_data():
    (x_train_full, y_train_full), (x_test,
                                   y_test) = keras.datasets.cifar10.load_data()

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_val = x_train_full[-10000:]
    y_val = y_train_full[-10000:]
    x_train = x_train_full[:-10000]
    y_train = y_train_full[:-10000]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_model():
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        data_augmentation,

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

#teste
def main():
    keras.utils.set_random_seed(42)

    print("Carregando e separando os dados...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_data()

    print("Construindo o modelo...")
    model = build_model()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    print("Iniciando o treinamento na CPU...")
    model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping]
    )

    print("\nCalculando métricas finais...")
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)

    print(f"\n========================================")
    print(f"Acurácia de Validação Final: {val_acc:.4f} ({val_acc * 100:.2f}%)")
    print(f"========================================\n")

    model_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "model.h5")
    model.save(model_path)
    print(f"Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()
