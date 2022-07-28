# CS-BGM

This repository provides code to reproduce results reported in the paper [Uncertainty Modeling in Generative Compressed Sensing](https://proceedings.mlr.press/v162/zhang22ai.html). 

Portions of the codebase in this repository uses codes originally provided in the open-source [CSGM](https://github.com/AshishBora/csgm) and  [Sparse-Gen](https://github.com/ermongroup/sparse_genSparse-Gen) repositories.


## Steps to reproduce the results
NOTE: Please run **all** commands from the root directory of the repository, i.e from `CS_BGM/`

### Requirements: 
---

1. Python 2.7
2. [Tensorflow 1.0.1](https://www.tensorflow.org/install/)
3. [Scipy](https://www.scipy.org/install.html)
4. [PyPNG](http://stackoverflow.com/a/31143108/3537687)
5. (Optional : for lasso-wavelet) [PyWavelets](http://pywavelets.readthedocs.io/en/latest/#install)
6. (Optional) [CVXOPT](http://cvxopt.org/install/index.html)
7. Other packages when running codes. Please install them when the requirements are reported.

Pip installation can be done by ```$ pip install -r requirements.txt```

### Preliminaries
---

1. Download/extract the datasets:

    ```shell
    $ ./setup/download_data.sh
    ```

2. The following command will unzip the pretrained model weights for the experiments:

   ```shell
   $ unzip models.zip
   ```

3. Download/extract pretrained models or train your own!

- To download pretrained models: ```$ ./setup/download_models.sh```
- To train your own
    - VAE on MNIST: ```$ ./setup/train_mnist_vae.sh```
    - DCGAN on celebA, see https://github.com/carpedm20/DCGAN-tensorflow

3. To use wavelet-based estimators, you need to create the basis matrix:

```shell
$ python ./src/wavelet_basis.py
```

### Demos
---

We provide some easy scripts in ``./quant_scripts/``. 

To obtain scripts to test the CS-BGM method, run

```shell
$ bash ./quant_scripts/celebA_bayesian.sh
```
for the fast implemention and

```shell
$ bash ./quant_scripts/celebA_VI.sh
```
for the complete version.


Then you will get a folder containing those scripts, say ``./scripts/``, next run

```shell
$ bash ./utils/run_sequentially.sh ./scripts/
```

To get results and make graphs, run

```shell
$ python ./src/make_graphs.py
```

### Supplementary
---

All parameters are carefully selected, while you can also fine-tune them to get better results. 

The core codes to implement CS-BGM can be found in the functions `dcgan_Baysian_estimator` and `dcgan_KL_estimator` of file `./src/celebA_estimators.py`. 

Please feel free to explore other codes in this repository. 

### Citation
---

```tex

@InProceedings{pmlr-v162-zhang22ai,
  title = 	 {Uncertainty Modeling in Generative Compressed Sensing},
  author =       {Zhang, Yilang and Xu, Mengchu and Mao, Xiaojun and Wang, Jian},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {26655--26668},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/zhang22ai/zhang22ai.pdf},
  url = 	 {https://proceedings.mlr.press/v162/zhang22ai.html},
}

```
