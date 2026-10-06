# MNIST privacy-preserving federated learning


## Run

Python 3.10 or later is recommended.

```powershell
python -m pip install -r requirements.txt
python download.py
python main.py
```

The run uses the cached MNIST dataset and writes per-round metrics to
`results/mnist_ppfrl_results.csv`. To check the full training and encryption
path quickly on a subset:

```powershell
python main.py --rounds 1 --max-train-samples 600 --client-delays 0 0 0
```

For the paper's 20 local epochs per round, pass `--local-epochs 20`. The default
uses one local epoch to keep a first full-dataset run practical. Other
paper-aligned defaults are three non-IID clients, 20 rounds, batch size 32,
learning rate 0.01, 8-bit update quantization, and a 128-bit Paillier key.
The key size, timer, and client delays are configurable; use `--help` for all
options.

## Implemented workflow

- MNIST is split into equal-sized label-skewed shards across three clients.
- Each round, clients train locally from the same global model and produce
  model deltas. A configurable virtual completion time models stragglers;
  updates arriving after the adaptive round deadline are dropped.
- Deltas are clipped and quantized to signed 8-bit integers. Several values
  are packed into each Paillier plaintext; the server adds ciphertexts without
  decrypting individual client updates.
- A fresh Paillier key pair is generated per round. The aggregate is decrypted
  to apply the mean update, and the global model is evaluated on the MNIST test
  set. Per-round accuracy, loss, timing, participating clients, and estimated
  encrypted traffic are recorded in the CSV.

## Scope and security notes

This is an experimental reproduction of the paper's workflow, not a
bit-for-bit reproduction: the paper does not fully specify its client timing
distribution or packed-integer encoding, and its “improved Paillier” equations
are not sufficiently clear to implement safely. This implementation therefore
uses the standard additive Paillier scheme and explicit, testable integer
packing. Client delays are simulated (defaults: 2, 6, and 12 seconds), not
measured network delays. The key holder and federated server are simulated in
one process; separating them and using threshold decryption would be required
for a deployment with a stronger trust boundary. The paper's 128-bit key size
is retained as a reproducibility setting, not a production security
recommendation.

Run the focused tests with:

```powershell
python -m unittest test_main.py
```
"# Minor-Project" 
