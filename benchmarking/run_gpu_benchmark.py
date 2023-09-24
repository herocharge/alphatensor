# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Reproducing GPU speedup reported in the AlphaTensor paper.

Showing GPU speedup of the provably correct fast matrix multiplication
algorithm discovered by AlphaTensor compared to the Strassen^2 baseline.

You should get around 8.5% speedup for multiplying matrices of size 8192 x 8192
in float32 precision on NVIDIA V100 GPU.

This code requires sudo access to set the GPU clock frequency (to reduce
benchmarking variance).

Ideally this code should be run on a server that is not used by others at the
same time to remove interference.

The code was tested on an n1-standard-96 Google Cloud VM instance with eight
V100 GPUs, Intel Skylake CPU, using the "TensorFlow Enterprise 2.9 (CUDA 11.3)"
image, and with Jax installed by running the following commands:
```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda]" \
  -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
"""
import subprocess

import numpy as np
# Might be needed on GCP because of the following bug:
# https://github.com/google/jax/issues/9218
import scipy.signal  # pylint: disable=unused-import

from alphatensor.benchmarking import factorizations
from alphatensor.benchmarking import utils
import matplotlib.pyplot as plt



def main():
  # process = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE)
  # output, _ = process.communicate()
  # if 'V100' not in str(output):
  #   raise ValueError('To reproduce the results from the paper, please run on a'
  #                    'server with V100 GPU.')
  # print('Fixing GPU clock frequency to 1530 to reduce benchmarking variance...')
  # process = subprocess.Popen(
  #     'sudo nvidia-smi -pm ENABLED -i 0'.split(' '), stdout=subprocess.PIPE)
  # output, _ = process.communicate()
  # process = subprocess.Popen(
  #     'sudo nvidia-smi --lock-gpu-clocks=1530,1530'.split(' '),
  #     stdout=subprocess.PIPE)
  # output, _ = process.communicate()
  # print('Done.')

  num_trials = 1
  matrix_sizes = [8192, 10240, 12288, 14336, 16384, 18432, 20480]
  matrix_sizes = [512]

  factorized_algorithms = [
      ('Strassen^2', factorizations.get_4x4x4_strassen_squared()),
      ('AlphaTensor GPU-optimized', factorizations.get_4x4x4_alphatensor_gpu()),
      ('AlphaTensor TPU-optimized', factorizations.get_4x4x4_alphatensor_tpu()),
  ]
  factorized_algorithms = np.load(open('alphatensor/benchmarking/alphatensor_14236_factorizations.npz', 'rb'))['factorizations']
  def reshaper(x):
    # return ('nn', np.reshape(x, (x.shape[1], x.shape[2], x.shape[0]), order='C'))
    new_x = []
    for i in range(3):
      z = []
      for j in range(16):
        y = []
        for k in range(49):
          y.append(x[k][i][j])
        z.append(np.array(y))
      new_x.append(np.array(z))
    return ('nn', np.array(new_x))
  factorized_algorithms = list(map(reshaper, factorized_algorithms))
  print((factorized_algorithms[0]))
  factorized_algorithms[0] = ('nn', factorizations.get_4x4x4_alphatensor_gpu())
  for s in matrix_sizes:
    print(f'Multiplying {s} x {s} matrices')
    print('='*40)
    results_dot = utils.benchmark_jnp_dot((s, s, s), num_trials=num_trials)
    i = 0
    speedups = []
    for algorithm_name, factorization in factorized_algorithms[:100]:
      # print(algorithm_name, factorization.shape)
      if algorithm_name == 'AlphaTensor TPU-optimized' and s > 19000:
        continue  # This TPU-optimized algorithm runs OOM on a V100 GPU.
      results_algorithm = utils.benchmark_factorized_algorithm(
          factorization, (s, s, s), num_trials=num_trials)
      ratio = np.median(results_dot / results_algorithm)
      improvement = 100 * ratio - 100
      speedups.append(improvement)
      print('%s%d vs `jnp.dot`: %0.2f%% speedup' % (algorithm_name, i+1, improvement))

    print('\n\n')

    plt.plot(range(len(speedups)), speedups)
    plt.show() 

if __name__ == '__main__':
  main()
