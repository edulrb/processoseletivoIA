import tensorflow_datasets as tfds
import tensorflow as tf
import os
import sys
import argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CLASS_NAMES = [
    "avião", "automóvel", "pássaro", "gato", "cervo",
    "cachorro", "sapo", "cavalo", "navio", "caminhão"
]


def load_cifar10_test():
    # Mesmo mirror (GCS via tensorflow-datasets) usado nos outros dois scripts.
    ds_test = tfds.load("cifar10", split="test",
                        batch_size=-1, as_supervised=True)
    x_test, y_test = tfds.as_numpy(ds_test)
    y_test = y_test.reshape(-1, 1)
    return x_test, y_test


def run_inference(tflite_path, num_samples=5):
    if not os.path.exists(tflite_path):
        print(f"Erro: Arquivo '{tflite_path}' não encontrado.")
        sys.exit(1)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    expected_dtype = input_details[0]['dtype']

    x_test, y_test = load_cifar10_test()

    np.random.seed(42)
    indices = np.random.choice(len(x_test), num_samples, replace=False)

    correct_count = 0

    print(f"Testando modelo em {num_samples} amostras:")

    for idx in indices:
        if expected_dtype == np.uint8 or expected_dtype == np.int8:
            input_data = x_test[idx].astype(expected_dtype)
        else:
            input_data = (x_test[idx] / 255.0).astype(expected_dtype)

        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        pred_idx = np.argmax(output_data)
        real_idx = y_test[idx][0]

        if pred_idx == real_idx:
            correct_count += 1

        print(
            f"Amostra {idx:04d} | Predito: {CLASS_NAMES[pred_idx]} | Real: {CLASS_NAMES[real_idx]}")

    print(f"\nAcertos: {correct_count}/{num_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.tflite")
    parser.add_argument("--samples", type=int, default=5)

    args = parser.parse_args()
    run_inference(args.model_path, args.samples)
