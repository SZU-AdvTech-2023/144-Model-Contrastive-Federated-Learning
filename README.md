# Model-Contrastive Federated Learning
原文 [Model-Contrastive Federated Learning](https://arxiv.org/pdf/2103.16257.pdf).

源码 [GitHub - QinbinLi/MOON: Model-Contrastive Federated Learning (CVPR 2021)](https://github.com/QinbinLi/MOON)

### Usage：

简单运行的例程

```python
python main.py --dataset=cifar10 \
    --device='cuda:0' \
    --model=simple-cnn \
    --alg=moon \
    --lr=0.01 \
    --mu=5 \
    --epochs=10 \
    --comm_round=100 \
    --n_parties=10 \
    --partition=noniid \
    --beta=0.5 \
    --logdir='./logs/' \
    --datadir='./data/'
```

### Dependencies：

```
PyTorch >= 1.0.0
torchvision >= 0.2.1
scikit-learn >= 0.23.1
seaborn >= 0.13.0
```

### Pre-Experiment：

预实验代码在**run.ipynb**中，使用jupyter notebook打开，在确认相关依赖安装完成后即可运行。

一部分为预测标签混淆矩阵热力图可视化，后一部分则是对隐藏向量抽取进行t-SNE降维可视化。

预实验部分由本人完成。


## Parameters

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model`                     | The model architecture. Options: `simple-cnn`, `resnet50` .|
| `alg` | The training algorithm. Options: `moon`, `fedavg`, `fedprox`, `local_training` |
| `dataset`      | Dataset to use. Options: `cifar10`. `cifar100`, `tinyimagenet`|
| `lr` | Learning rate. |
| `batch-size` | Batch size. |
| `epochs` | Number of local epochs. |
| `n_parties` | Number of parties. |
| `sample_fraction` | the fraction of parties to be sampled in each round. |
| `comm_round`    | Number of communication rounds. |
| `partition` | The partition approach. Options: `noniid`, `iid`. |
| `beta` | The concentration parameter of the Dirichlet distribution for non-IID partition. |
| `mu` | The parameter for MOON and FedProx. |
| `temperature` | The temperature parameter for MOON. |
| `out_dim` | The output dimension of the projection head. |
| `datadir` | The path of the dataset. |
| `logdir` | The path to store the logs. |
| `device` | Specify the device to run the program. |
| `seed` | The initial seed. |
