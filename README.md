# SNIP
This repository contains code for the Masterproof [Single-shot pruning van convolutionele netwerken].

## Prerequisites

### Dependencies
* imgaug==0.4.0
* tensorflow==1.15.0
* sklearn==0.0
* numpy==1.18.1
* pandas==1.0.3

### Datasets
Put the following datasets in your preferred location (e.g., `./`).
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [MNIST-FASHION](https://github.com/zalandoresearch/fashion-mnist)
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

## Usage
To run the code (LeNet on MNIST by default):
```
python main.py --path_data=./
```
See `main.py` to run with other options.

## Citation
If you use this code for your work, please cite the following:
```
@inproceedings{
  title={Single-shot pruning van convolutionele netwerken},
  author={Puype, Jurriën},
  school={University of Ghent},
  year={2020},
  month={6},
}
```

## License
This project is licensed under the GNU GENERAL PUBLIC LICENSE.
See the [LICENSE](https://github.com/jbpuype/sSNIP-SP/blob/master/LICENSE) file for details.
