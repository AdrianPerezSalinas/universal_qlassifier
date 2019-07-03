# Code for *[Data re-uploading for a universal quantum classifier](https://arxiv.org/abs/1806.11463)*
#### Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and José I. Latorre.

This is a repository for all code written for the article "*Data re-uploading for a universal quantum classifier*. Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and José I. Latorre. [arXiv: [quant-ph]](https://arxiv.org/abs/)."
It gives numerical simulations of the [quantum classifier] in.

All code is written Python. Libraries required:
  - matplotlib for plots
  - numpy, os, scipy
  - scikit-learn

##### Files included:
  - [QuantumState.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/QuantumState.py): Simulator of a quantum circuit using only basic Python packages such as numpy
  - [big_functions.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/big_functions.py): Functions acting as the master of all other subroutines in the simulator
  - [circuitery.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/circuitery.py): Translates the problem to the quantum circuit basic level.
  - [classical_benchmark.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/classical_benchmark.py): Provides some classical examples using scikit learn.
  - [data_gen.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/data_gen.py): Generates random training and data set for different problems.
  - [fidelity_minimization.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/fidelity_minimization.py): All the code needed for the fidelity cost function.
  - [main.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/main.py): This is the only file one needs to change. Everything can be set up there: number of qubits, layers, entanglement, cost function, problem, etc. The only thing one has to do is to run this file.
  - [problem__gen.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/problem_gen.py): Generates data of the problem we need for other files.
  - [save_data.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/save_data.py): Saves results in text files and images. 
  - [test_data.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/test_data.py): Tests the performance of the classifier, and outputs variables needed for saving data.
  - [weighted_fidelity_minimization.py](https://github.com/AdrianPerezSalinas/universal_qlassifier/blob/master/weighted_fidelity_minimization.py): All the code needed for the weighted fidelity cost function.
##### How to cite
If you use this code in your research, please cite it as follows:

A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster and J.I. Latorre, "Data re-uploading for a universal quantum classifier", (2019), arXiv: [quant-ph], https://arxiv.org/abs/

BibTeX:
```
@misc{PerezSalinas2019DRUUQC,
  author = {Pérez-Salinas, A. and Cervera-Lierta, A. and Gil-Fuster, E. and Latorre, J. I.},
  title = {{Data} re-uploading for a universal quantum classifier},
  year = {2019},
  journal={arXiv:},
  url = {https://arxiv.org/abs/}
}
```

