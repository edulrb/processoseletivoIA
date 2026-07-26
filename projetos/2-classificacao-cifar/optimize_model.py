import sys
import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def evaluate_tflite_model(interpreter, x_test, y_test):
    """
    Executa inferência sequencial no modelo TFLite para medir a acurácia final.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    correct_predictions = 0
    total_images = len(x_test)

    for i in range(total_images):
        input_data = np.expand_dims(x_test[i], axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        if np.argmax(output_data) == y_test[i][0]:
            correct_predictions += 1

        # Log discreto de sobrevivência para evitar a impressão de que o terminal travou
        if (i + 1) % 2500 == 0:
            print(f"Inferência TFLite: {i + 1}/{total_images}")

    return correct_predictions / total_images


def optimize_and_convert(model_path, output_path):
    """
    Converte um modelo Keras para TFLite usando Dynamic Range Quantization 
    (pesos em int8, ativações em float32) e compara o desempenho.
    """
    if not os.path.exists(model_path):
        print(f"Erro: Arquivo '{model_path}' não encontrado.")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0

    _, original_acc = model.evaluate(x_test, y_test, verbose=0)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    tflite_acc = evaluate_tflite_model(interpreter, x_test, y_test)

    h5_size = os.path.getsize(model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(output_path) / (1024 * 1024)

    print("\n--- Relatório de Otimização ---")
    print(f"Tamanho .h5:       {h5_size:.2f} MB")
    print(f"Tamanho .tflite:   {tflite_size:.2f} MB")
    print(f"Acurácia .h5:      {original_acc:.4f}")
    print(f"Acurácia .tflite:  {tflite_acc:.4f}")
    print("-------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Otimiza modelo Keras para TensorFlow Lite.")
    parser.add_argument("--model_path", type=str,
                        default="model.h5", help="Caminho do modelo original")
    parser.add_argument("--output_path", type=str,
                        default="model.tflite", help="Caminho do modelo otimizado")

    args = parser.parse_args()
    optimize_and_convert(args.model_path, args.output_path)
