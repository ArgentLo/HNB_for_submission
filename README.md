# HANABI: Graph Embedding for Recommendation via Conditional Proximity

This is our implementation for the paper.

In this repository, all the following parts are included:
  - **all datasets** used in the paper
  - **all baseline models** for comparison
  - the proposed **HNB model**
  - **quick start** instruction for training & evaluation

----

### Double-Blind Submission

In order to have a fair and respectful submission, the **Implementation is Anonymized** with all authors' information removed.


### Introduction

HANABI (HNB) is a recommendation framework based on graph embedding collaborative filtering, conditionally encoding the imformation flowing from high-order end nodes through intermediate nodes to source nodes.

There are 7 baseline models used for comparative purpose in our experiments. To help reproduce the results on the paper, we summarized all models as follows:

- Matrix Factorization (MF) : [Implementaion by LightFM](https://github.com/lyst/lightfm)
- Bayesian Pairwise Reranking MF (BPR) : **included in this repository**.
- Neural Matrix Factorization (NMF) : https://github.com/hexiangnan/neural_collaborative_filtering
- Collaborative Memory Network (CMN) : https://github.com/tebesu/CollaborativeMemoryNetwork

- High-Order Proximity Recommendation (HOP-Rec) : https://github.com/cnclabs/smore
- Collaborative Similarity Embedding (RANKCSE) : **included in this repository**.
- Neural Graph Collaborative Filtering (NGCF) : https://github.com/xiangwang1223/neural_graph_collaborative_filtering

- HANABI: Conditional Proximity Graph Embedding (HNB) : **included in this repository**.

----

### Environment Requirement

The code has been tested running under Python 3.5. The required packages are as follows:

```
tensorflow == 1.12.0
numpy
tqdm
networkx
torch
```

----

### Dataset

All datasets used in the experiments are provided. 

Since implicit feedback data are considered in our work, all **data values are binarized**. 

For all dataset, 80% of a user’s historical items would be randomly sampled as the training set and the rest items are collected as the test set.

Please **download** the preprocessed datasets and **save in `./data/`**.

- Citeulike-a (19M) :

  ```
  https://drive.google.com/open?id=1mW5UD8Ds29fN0lH9JcvBuf-yAg_ZYdWl
  ```

- MovieLens-1M (76M):

  ```
  https://drive.google.com/open?id=1rwGV60iK_Cqtx82J3DoV0IMMA8seaMBq
  ```

- Gowalla (96M):

  ```
  https://drive.google.com/open?id=1PgkmhbbDJODgCJJBM82eXN3xgoWV9bxh
  ```

- Amazon-Book (277M):

  ```
  https://drive.google.com/open?id=1m_RbYu_iRODtpOQXGO_ymfUYuDxOt2DS
  ```

----

### Evaluation

To save your training time, checkpoints of each model on various datasets are provided.

Please **download** the embedding checkpoints and **save in `./saved_embeddings/`**.

#### Checkpoints:

- **BPR Embedding**:

  ```
  Citeulike-a: 
  https://drive.google.com/file/d/1544tCAqP_YcL07EW_lkmls0vkT5UPoa4
  ML-1M: 
  https://drive.google.com/file/d/1jAG2uKAMKoovfiszm05A0YRX2D07vJWY
  Gowalla: 
  https://drive.google.com/file/d/16LmFg5-2s1ttiOY0oHnZ9_mOs97Xz2wm
  Amazon-Book: 
  https://drive.google.com/file/d/1f4JCAkAUbKK9mEGIXiglBMEoU1Wi13eE
  ```

- **CSE Embedding**:

  ```
  Citeulike-a: 
  https://drive.google.com/open?id=148Jhnl0YGfw1UnP4xy-1-muGpPHFBTrh
  ML-1M: 
  https://drive.google.com/open?id=1XlwI7qQP2iODG37xvaF1Ce7Xl4gv1LSc
  Gowalla: 
  https://drive.google.com/open?id=1SZnQj8VOtg0O6VikINn6vtCSXcX2LgVf
  Amazon-Book: 
  https://drive.google.com/open?id=1egqL49NhDk4ddAhNDjydFdGzEONm5-XO
  ```

- **HNB Embedding**:

  ```
  Citeulike-a: 
  https://drive.google.com/open?id=16ke-aMlzLGt7dbdl686PJtWjC1gcviSf
  ML-1M: 
  https://drive.google.com/open?id=1xjBLdpI_qcWZtUjtyAKKjAAuCAQbPJzf
  Gowalla: 
  https://drive.google.com/open?id=1p2Y38X7YE0izh55tNVlBJYBdHWMbMrvb
  Amazon-Book: 
  https://drive.google.com/open?id=14ljCSSIdMLeyqNg-xPz__AUHuUnt5e7h
  ```

#### Evaluation Example:

```sh
python3 ./eval_files/eval_AMZ_B.py --saved_emb emb_name --emb_size 100 --top_k 20

# eval_Citeulike.py
# eval_Gowalla.py
# eval_AMZ_B.py
# eval_ML1M.py
```

----

### Training Example

Here we provide examples of training HNB on different dataset. Note that, by simply removing the `--eta` and replacing with desired model, the configuration can also be applied in training `BPR` and `CSE`.

- Citeulike-a Training:

  ```
  python3 main_hnb_2hop.py  --batch_size 4096 --epoch 150 --dataset citeulike --eta 3 --embed_size 100 --save_epoch 5
  ```

- MovieLens-1M Training:

  ```
  python3 main_hnb_2hop.py  --batch_size 4096 --epoch 100 --dataset ML-1M --eta 3 --embed_size 100 --save_epoch 5
  ```

- Gowalla Training:

  ```
  python3 main_hnb_2hop.py  --batch_size 4096 --epoch 100 --dataset gowalla --eta 3 --embed_size 100 --save_epoch 5
  ```

- Amazon-Book Training:

  ```
  python3 main_hnb_2hop.py  --batch_size 4096 --epoch 100 --dataset amzbook --eta 3 --embed_size 100 --save_epoch 5
  ```
