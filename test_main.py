import unittest

import numpy as np
from phe import paillier

from main import (
    decrypt_packed_average,
    encrypt_packed_updates,
    partition_non_iid,
    quantize_update,
)


class FederatedLearningTests(unittest.TestCase):
    def test_quantization_clips_and_uses_signed_eight_bit_range(self):
        quantized = quantize_update(np.array([-2.0, -0.5, 0.0, 0.5, 2.0]), 1.0)

        np.testing.assert_array_equal(quantized, [-127, -64, 0, 64, 127])

    def test_encrypted_packed_average_matches_plaintext_average(self):
        public_key, private_key = paillier.generate_paillier_keypair(n_length=128)
        updates = [
            np.array([-127, 5, 0, 126, -1], dtype=np.int16),
            np.array([127, -3, 1, 0, -1], dtype=np.int16),
            np.array([1, 0, -1, 0, 1], dtype=np.int16),
        ]

        encrypted, vector_length = encrypt_packed_updates(updates, public_key)
        actual = decrypt_packed_average(
            encrypted, private_key, vector_length, len(updates)
        )
        expected = np.mean(np.stack(updates), axis=0)

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_encryption_rejects_values_that_could_overflow_int16(self):
        public_key, _ = paillier.generate_paillier_keypair(n_length=128)

        with self.assertRaisesRegex(ValueError, "signed range"):
            encrypt_packed_updates([np.array([65536])], public_key)

    def test_non_iid_partition_is_complete_and_reproducible(self):
        labels = np.repeat(np.arange(10), 30)

        first = partition_non_iid(labels, client_count=3, seed=7)
        second = partition_non_iid(labels, client_count=3, seed=7)

        self.assertEqual([len(indices) for indices in first], [100, 100, 100])
        np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(300))
        for first_client, second_client in zip(first, second):
            np.testing.assert_array_equal(first_client, second_client)


if __name__ == "__main__":
    unittest.main()
