# Graph Convolutional Networks

## Solutions to Part 2 - GCN

### Base class

Base class is implemented in `message_passing.py`. It builds on top of `torch.nn.Module` and has four separate methods: aggregate, initialize, update, output as mentioned in question.

### Implementation of GCN

```bash
$ python -m src.main --task gcn --epochs 100
```

My implementation gets the following results: `val_loss=1.47, train_loss=1.05, val_accuracy=0.618, train_accuracy=1` on CiteSeer dataset. The implementation is given in `gcn.py` file, it builds on top of `message_passing.py` as required in question.

The implementation is according to the assigned paper: ["Semi-Supervised Classification with Graph Convolutional Networks"](https://arxiv.org/abs/1609.02907). We have one "encoder"-like NN that converts feature vectors from input dimension to our internal latent dimension. Then we apply the $l$-layers of our GNN in the latent dimension using message passing. Then we convert back from the latent dimension to the final output dimension, which is six labels for the classification task.


#### Performance and reason for difference

1. No normalization:
2. Row normalization:
3. Column normalization:
4. Symmetric normalization:

### GCN comparison with Graph Isomorphism Network (GIN)

GIN is implemented according to the formula given in this paper: ["How Powerful are Graph Neural Networks?"](https://arxiv.org/abs/1810.00826).

$$h_v^{(k)}=\textrm{MLP}^{(k)}\left(\left(1 + \epsilon^{(k)}\right)\cdot h_v^{(k - 1)}+\sum_{u\in\mathcal{N}(v)}h_u^{(k-1)}\right)$$


### Vanilla RNN code

The IMDB dataset is sourced from `torchtext.datasets` which itself sources the dataset from the ACL 2011 paper: ["Learning Word Vectors for Sentiment Analysis"](https://ai.stanford.edu/~amaas/data/sentiment/) ACL 2011 by Andrew et. al.

## Requirements

* torch (1.11)
* numpy (1.25)
* scikit-learn (1.8)
* torchtext
* tqdm