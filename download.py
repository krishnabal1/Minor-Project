"""Ensure that the MNIST dataset used by main.py is available in the Keras cache."""

from tensorflow.keras.datasets import mnist


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(
        f"MNIST ready: {len(train_images):,} training images, "
        f"{len(test_images):,} test images, {len(set(train_labels))} classes"
    )