# `similaritybenchmark`

Python 3 port of the original [`similaritybenchmark`](https://github.com/nextmovesoftware/similaritybenchmark) in support of the `CheMeleon` study.

## Usage

This code was run using Python 3.12 but should work with all modern python versions that support the required dependencies (which are listed in their respective sections).

### Preparing the Data

Run `python decompress.py` (on first clone/download only) to decompress the benchmarking data.

Requirements: None

### Running the Benchmark

The original set of fingerprints considered has been significantly cut down from the original study to focus more on 'classes' of fingerprints rather than their hyperparameters.
You can control which fingerprints are tested (and implement your own) in `benchlib/fingerprint_lib.py` by commenting out lines in the `FP_LOOKUP` dictionary.
This is **especially** important when it comes to multiprocessing.
You can set `ENABLE_PARALLEL` in your shell to run the benchmark on all available CPUs, but this doesn't work with CUDA and pytorch.

Run `python run_benchmark.py` to run the benchmark.

Requirements: `rdkit tqdm psutil scipy numpy chemprop~=2.1.0`

This file was refactored quite a bit from the original study, most especially for adding parallelism.

Running the classical fingerprints takes about 9 hours on 8 physical CPU cores.
Running the `chemeleon` fingerprint takes 18 hours with a laptop NVIDIA GPU.

### Analyzing the Results

Run `python analyze_results.py`

Requirements: `numpy scipy tqdm`

This file is mostly unchanged from the original study.

This should run in O(minutes).

Run `projections.ipynb` to generate the visualization of the learned space for `CheMeleon`.
