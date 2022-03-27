# Graph Convolutional Networks

Question 3 of the assignment

## Solutions to Part 2 - GCN

### Base class

Base class is implemented in `message_passing.py`. It builds on top of `torch.nn.Module` and has four separate methods: aggregate, initialize, update, output as mentioned in question.

### Implementation of GCN

```bash
$ cd q3
$ python -m src.main --task gcn --epochs 100
```

My implementation gets the following results: `val_loss=1.47, train_loss=1.05, val_accuracy=0.618, train_accuracy=1` on CiteSeer dataset. The implementation is given in `gcn.py` file, it builds on top of `message_passing.py` as required in question.

The implementation is according to the assigned paper: ["Semi-Supervised Classification with Graph Convolutional Networks"](https://arxiv.org/abs/1609.02907). We have one "encoder"-like NN that converts feature vectors from input dimension to our internal latent dimension. Then we apply the $l$-layers of our GNN in the latent dimension using message passing. Then we convert back from the latent dimension to the final output dimension, which is six labels for the classification task.

### GCN comparison with Graph Isomorphism Network (GIN)

GIN is implemented according to the formula given in this paper: ["How Powerful are Graph Neural Networks?"](https://arxiv.org/abs/1810.00826).

$$h_v^{(k)}=\textrm{MLP}^{(k)}\left(\left(1 + \epsilon^{(k)}\right)\cdot h_v^{(k - 1)}+\sum_{u\in\mathcal{N}(v)}h_u^{(k-1)}\right)$$

## Installation

```bash
python setup.py install
```

## Requirements

* torch (1.11)
* numpy (1.25)
* scikit-learn (1.8)

## Run the code

```bash
scripts/download.sh
python -m src.trainer --epochs 100 # for part 1
python -m src.trainer # for citeseer training
```



## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), 
* an N by D feature matrix (D is the number of features per node), and
* an N by E binary label matrix (E is the number of classes).

Have a look at the `load_data()` function in `utils.py` for an example.

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://www.cs.umd.edu/~sen/lbc-proj/LBC.html. In our version (see `data` folder) we use dataset splits provided by https://github.com/kimiyoung/planetoid (Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861), ICML 2016). 

You can specify a dataset as follows:

```bash
python train.py --dataset citeseer
```

(or by editing `train.py`)

## Models

You can choose between the following models: 
* `gcn`: Graph convolutional network (Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907), 2016)
* `gcn_cheby`: Chebyshev polynomial version of graph convolutional network as described in (Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), NIPS 2016)
* `dense`: Basic multi-layer perceptron that supports sparse inputs

## Graph classification

Our framework also supports batch-wise classification of multiple graph instances (of potentially different size) with an adjacency matrix each. It is best to concatenate respective feature matrices and build a (sparse) block-diagonal matrix where each block corresponds to the adjacency matrix of one graph instance. For pooling (in case of graph-level outputs as opposed to node-level outputs) it is best to specify a simple pooling matrix that collects features from their respective graph instances, as illustrated below:

![graph_classification](https://user-images.githubusercontent.com/7347296/34198790-eb5bec96-e56b-11e7-90d5-157800e042de.png)


## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```