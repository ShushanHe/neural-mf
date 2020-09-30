# Network Diffusions via Neural Mean-Field Dynamics

This repository is the official implementation of [NMF](https://arxiv.org/abs/2006.09449) model in Tensorflow. 

NMF model is a learning framework based on neural mean-field dynamics for inference and estimation probleams of diffusion on networks. 
Directly using cascade data, NMF model can jointly learn the structure of the diffusion network and the evolution of infection probabilities.

## Requirements

We conducted experiments under

* Python 3.7.7

* Tensorflow 1.13

* cuda 9.1 

## Usage
***Example Usage***

* To generate train data, run this command:

	`python train_data_gene.py ` 

* To train and test the model in the paper, run this command:

	`python NMF_tf.py ` 


* To present results as numpy matrixs, run this command:

	`python Outputs.py ` 


## Citation
Please cite our paper if you use this code in your research:
```
@inproceedings{he2020NMF,
  title={Network Diffusions via Neural Mean-Field Dynamics},
  author={He, Shushan and Zha, Hongyuan and Ye, Xiaojing},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
  location{Vancouver, Canada}
  numpages={9}
}
```

## License

Code released under the [MIT License](LICENSE).
