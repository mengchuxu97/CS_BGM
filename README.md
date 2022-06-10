# CS-BGM

This repository provides code to reproduce results (Paper: CS-BGM). 

Portions of the codebase in this repository uses codes originally provided in the open-source [CSGM](https://github.com/AshishBora/csgm) and  [Sparse-Gen](https://github.com/ermongroup/sparse_genSparse-Gen) repositories.


## Steps to reproduce the results
NOTE: Please run **all** commands from the root directory of the repository, i.e from ```CS_BGM/```

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

2. The following command will unzip the trained model weights for the experiments:

   ```shell
   $ unzip models.zip
   ```

3. Download/extract pretrained models or train your own!

- To download pretrained models: ```$ ./setup/download_models.sh```
- To train your own
    - VAE on MNIST: ```$ ./setup/train_mnist_vae.sh```
    - DCGAN on celebA, see https://github.com/carpedm20/DCGAN-tensorflow

3. To use wavelet based estimators, you need to create the basis matrix:

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


Then you will get a folder containing those scripts, say ``./scripts/``, then run

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

The core codes to implement CS-BGM are the functions `dcgan_Baysian_estimator` and `dcgan_KL_estimator` in file `./src/celebA_estimators.py`. 

Please feel free to explore other codes in this repository. 

### Citation
---

```tex

@InProceedings{CS-BGM,
  title = 	 {Modeling Uncertainties in Generative Compressed Sensing},
  author =       {Yilang Zhang and Mengchu Xu and Xiaojun Mao and Jian Wang},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {},
  year = 	 {2022},
  volume = 	 {},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
}

```
