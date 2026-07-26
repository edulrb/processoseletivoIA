import os
import sys
import argparse
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def run_inference(tflite_path, num_samples=5):
    if not os.path.exists(tflite_path):
        print(f"Erro: Arquivo '{tflite_path}' não encontrado.")
        sys.exit(1)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # dtype pode mudar dependendo da quantização usada (float32 no dynamic range,
    # mas seria uint8/int8 se fosse full-integer)
    expected_dtype = input_details[0]['dtype']

    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    np.random.seed(42)  # fixo só pra eu conseguir comparar execuções diferentes
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

        print(f"Amostra {idx:04d} | Predito: {CLASS_NAMES[pred_idx]} | Real: {CLASS_NAMES[real_idx]}")

    print(f"\nAcertos: {correct_count}/{num_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.tflite")
    parser.add_argument("--samples", type=int, default=5)

    args = parser.parse_args()
    run_inference(args.model_path, args.samples)