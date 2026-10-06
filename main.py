"""MNIST experiment for privacy-preserving asynchronous federated learning."""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
from phe import paillier


CLIENT_COUNT = 3
QUANTIZATION_BITS = 8
QUANTIZATION_LIMIT = (1 << (QUANTIZATION_BITS - 1)) - 1


def quantize_update(update: np.ndarray, clip_value: float) -> np.ndarray:
    """Clip and map a model delta to signed 8-bit integer levels."""
    if clip_value <= 0:
        raise ValueError("clip_value must be greater than zero")
    values = np.asarray(update, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("Model updates must contain finite values")
    clipped = np.clip(values, -clip_value, clip_value)
    return np.rint(clipped * (QUANTIZATION_LIMIT / clip_value)).astype(np.int16)


def partition_non_iid(
    labels: np.ndarray, client_count: int, seed: int
) -> list[np.ndarray]:
    """Split label-sorted data into equal shards and randomly assign them."""
    if client_count < 1:
        raise ValueError("client_count must be at least one")
    shard_count = client_count * 10
    if len(labels) < shard_count:
        raise ValueError(
            f"At least {shard_count} training examples are required to create "
            f"{client_count} clients"
        )

    ordered_indices = np.argsort(labels, kind="stable")
    shards = [shard for shard in np.array_split(ordered_indices, shard_count)]
    random.Random(seed).shuffle(shards)
    return [
        np.concatenate(shards[client_id::client_count])
        for client_id in range(client_count)
    ]


def encrypt_packed_updates(
    updates: Sequence[np.ndarray],
    public_key: paillier.PaillierPublicKey,
    quantization_limit: int = QUANTIZATION_LIMIT,
) -> tuple[list[paillier.EncryptedNumber], int]:
    """Pack quantized client deltas into Paillier plaintexts and encrypt them."""
    if not updates:
        raise ValueError("At least one client update is required")
    if quantization_limit < 1:
        raise ValueError("quantization_limit must be at least one")

    flattened = []
    for update in updates:
        raw_update = np.asarray(update)
        if not np.issubdtype(raw_update.dtype, np.number) or not np.all(
            np.isfinite(raw_update)
        ):
            raise ValueError("Client updates must contain finite numeric values")
        if np.any(raw_update != np.rint(raw_update)):
            raise ValueError("Client updates must contain quantized integer values")
        if np.any(raw_update < -quantization_limit) or np.any(
            raw_update > quantization_limit
        ):
            raise ValueError("Quantized updates exceed the configured signed range")
        flattened.append(raw_update.astype(np.int16).reshape(-1))

    vector_length = len(flattened[0])
    if any(len(update) != vector_length for update in flattened):
        raise ValueError("All client updates must have the same number of values")

    slot_width = math.ceil(
        math.log2(2 * quantization_limit * len(flattened) + 1)
    )
    slots_per_ciphertext = (public_key.n.bit_length() - 1) // slot_width
    if slots_per_ciphertext < 1:
        raise ValueError("Paillier key is too small to pack an update")

    encrypted_sums: list[paillier.EncryptedNumber] = []
    for start in range(0, vector_length, slots_per_ciphertext):
        packed_clients = []
        for update in flattened:
            packed = 0
            for slot, value in enumerate(
                update[start : start + slots_per_ciphertext]
            ):
                packed |= (int(value) + quantization_limit) << (slot * slot_width)
            packed_clients.append(public_key.encrypt(packed))

        encrypted_sum = packed_clients[0]
        for encrypted_client_update in packed_clients[1:]:
            encrypted_sum += encrypted_client_update
        encrypted_sums.append(encrypted_sum)

    return encrypted_sums, vector_length


def decrypt_packed_average(
    encrypted_sums: Sequence[paillier.EncryptedNumber],
    private_key: paillier.PaillierPrivateKey,
    vector_length: int,
    client_count: int,
    quantization_limit: int = QUANTIZATION_LIMIT,
) -> np.ndarray:
    """Decode the homomorphic sum and return its signed integer average."""
    if client_count < 1:
        raise ValueError("client_count must be at least one")
    if vector_length < 0:
        raise ValueError("vector_length cannot be negative")

    slot_width = math.ceil(
        math.log2(2 * quantization_limit * client_count + 1)
    )
    slots_per_ciphertext = (private_key.public_key.n.bit_length() - 1) // slot_width
    expected_ciphertexts = math.ceil(vector_length / slots_per_ciphertext)
    if len(encrypted_sums) != expected_ciphertexts:
        raise ValueError(
            f"Encrypted update has {len(encrypted_sums)} ciphertexts; "
            f"expected {expected_ciphertexts}"
        )
    slot_mask = (1 << slot_width) - 1
    values = np.empty(vector_length, dtype=np.float32)

    offset = 0
    for encrypted_sum in encrypted_sums:
        packed_sum = private_key.decrypt(encrypted_sum)
        remaining = vector_length - offset
        slots_in_ciphertext = min(slots_per_ciphertext, remaining)
        for slot in range(slots_in_ciphertext):
            summed_unsigned = (packed_sum >> (slot * slot_width)) & slot_mask
            values[offset] = (
                summed_unsigned - client_count * quantization_limit
            ) / client_count
            offset += 1

    if offset != vector_length:
        raise ValueError(
            f"Encrypted update has {offset} values; expected {vector_length}"
        )
    return values


def create_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(8, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(16, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def load_mnist(
    max_train_samples: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )
    train_images = train_images.astype(np.float32)[..., np.newaxis] / 255.0
    test_images = test_images.astype(np.float32)[..., np.newaxis] / 255.0
    train_labels = train_labels.astype(np.int64)
    test_labels = test_labels.astype(np.int64)

    if max_train_samples is not None:
        if max_train_samples < CLIENT_COUNT * 10:
            raise ValueError(
                f"--max-train-samples must be at least {CLIENT_COUNT * 10}"
            )
        rng = np.random.default_rng(seed)
        selected = rng.permutation(len(train_images))[:max_train_samples]
        train_images = train_images[selected]
        train_labels = train_labels[selected]

    return train_images, train_labels, test_images, test_labels


def run_experiment(args: argparse.Namespace) -> list[dict[str, float | int]]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    train_images, train_labels, test_images, test_labels = load_mnist(
        args.max_train_samples, args.seed
    )
    client_indices = partition_non_iid(train_labels, CLIENT_COUNT, args.seed)
    client_delays = args.client_delays
    if len(client_delays) != CLIENT_COUNT or any(delay < 0 for delay in client_delays):
        raise ValueError("--client-delays must contain three non-negative values")

    global_model = create_model()
    timer_seconds = args.initial_timer
    history: list[dict[str, float | int]] = []

    print(
        f"MNIST: {len(train_images):,} training examples, "
        f"{len(test_images):,} test examples; {CLIENT_COUNT} non-IID clients"
    )
    print(
        f"Rounds={args.rounds}, local epochs={args.local_epochs}, "
        f"Paillier key={args.key_bits} bits, initial timer={timer_seconds:g}s",
        flush=True,
    )

    for round_index in range(args.rounds):
        round_started = time.perf_counter()
        global_weights = global_model.get_weights()
        client_updates: list[np.ndarray] = []
        received_client_ids: list[int] = []

        # Client completion times are simulated so straggler behavior is
        # repeatable and does not depend on the machine running the experiment.
        for client_id, indices in enumerate(client_indices):
            if client_delays[client_id] > timer_seconds:
                continue

            local_model = create_model()
            local_model.set_weights(global_weights)
            local_model.fit(
                train_images[indices],
                train_labels[indices],
                batch_size=args.batch_size,
                epochs=args.local_epochs,
                verbose=0,
                shuffle=True,
            )
            client_weights = local_model.get_weights()
            delta = np.concatenate(
                [
                    (client_weight - global_weight).reshape(-1)
                    for client_weight, global_weight in zip(
                        client_weights, global_weights
                    )
                ]
            )
            client_updates.append(quantize_update(delta, args.clip_value))
            received_client_ids.append(client_id)

        if not client_updates:
            raise RuntimeError(
                f"No client update arrived before the {timer_seconds:g}s "
                f"deadline in round {round_index + 1}; increase --initial-timer"
            )

        key_started = time.perf_counter()
        public_key, private_key = paillier.generate_paillier_keypair(
            n_length=args.key_bits
        )
        encrypted_sums, vector_length = encrypt_packed_updates(
            client_updates, public_key
        )
        encryption_seconds = time.perf_counter() - key_started

        aggregation_started = time.perf_counter()
        mean_quantized_delta = decrypt_packed_average(
            encrypted_sums,
            private_key,
            vector_length,
            len(client_updates),
        )
        aggregation_seconds = time.perf_counter() - aggregation_started

        dequantized_delta = (
            mean_quantized_delta * (args.clip_value / QUANTIZATION_LIMIT)
        )
        updated_weights: list[np.ndarray] = []
        offset = 0
        for weight in global_weights:
            size = weight.size
            layer_delta = dequantized_delta[offset : offset + size].reshape(
                weight.shape
            )
            updated_weights.append(weight + layer_delta.astype(weight.dtype))
            offset += size
        global_model.set_weights(updated_weights)

        loss, accuracy = global_model.evaluate(
            test_images, test_labels, batch_size=args.batch_size, verbose=0
        )
        ciphertext_bytes = (
            len(encrypted_sums)
            * ((public_key.nsquare.bit_length() + 7) // 8)
            * (len(client_updates) + 1)
        )
        result: dict[str, float | int] = {
            "round": round_index + 1,
            "accuracy": float(accuracy),
            "loss": float(loss),
            "accepted_clients": len(received_client_ids),
            "timer_seconds": timer_seconds,
            "encryption_seconds": encryption_seconds,
            "aggregation_seconds": aggregation_seconds,
            "round_seconds": time.perf_counter() - round_started,
            "communication_bytes": ciphertext_bytes,
            "key_bits": args.key_bits,
            "quantization_bits": QUANTIZATION_BITS,
        }
        history.append(result)
        print(
            f"Round {round_index + 1:02d}/{args.rounds}: "
            f"accuracy={accuracy:.4f}, loss={loss:.4f}, "
            f"clients={len(received_client_ids)}/{CLIENT_COUNT}, "
            f"timer={timer_seconds:g}s, "
            f"encrypted traffic~{ciphertext_bytes / 1024:.1f} KiB",
            flush=True,
        )

        timer_seconds += args.timer_increment

    if args.results:
        output_path = Path(args.results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as results_file:
            writer = csv.DictWriter(results_file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"Round metrics saved to {output_path.resolve()}")

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-inspired encrypted asynchronous MNIST FL experiment."
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--key-bits", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument("--clip-value", type=float, default=1.0)
    parser.add_argument("--initial-timer", type=float, default=10.0)
    parser.add_argument("--timer-increment", type=float, default=5.0)
    parser.add_argument(
        "--client-delays",
        type=float,
        nargs=CLIENT_COUNT,
        default=(2.0, 6.0, 12.0),
        metavar=("CLIENT_1", "CLIENT_2", "CLIENT_3"),
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", default="results/mnist_ppfrl_results.csv")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be at least one")
    if args.local_epochs < 1 or args.batch_size < 1:
        parser.error("--local-epochs and --batch-size must be at least one")
    if args.initial_timer < 0 or args.timer_increment < 0:
        parser.error("timer values cannot be negative")
    if args.clip_value <= 0:
        parser.error("--clip-value must be greater than zero")
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
