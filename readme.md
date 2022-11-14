# DASFAA2023 463

This is the official implementation of the submission *Few-shot Link Prediction for Event-based Social Networks via Meta-learning*.

#### Requirements
pytorch=1.8.0

dgl-cu102=0.6.1

prettytable

scikit-learn

pandas

#### Datasets
We implement the proposed framework on both DBLP and Tmall. 

Since the DBLP dataset is collected from publicly available website, we have released the processed DBLP data (5 folds) in the repository. To follow the compliance policies, we will release the Tmall dataset upon the acceptance.

#### Commands

We specify the commands for DBLP dataset as follows:

For overall performance:

```
python main.py --dataset dblp_1025 --fold 2 --valid_freq 50 --update_step_test 201 --train_ratio 0.3
```

For fast adaption:

```
python main.py --dataset dblp_1025 --fold 2 --fast_adaption_flag --valid_freq 20 --train_ratio 0.3
```

Notably, you can refer to the Argument Parser module to define other hyperparameters, e.g., the support ratio, auxiliary learning trade-off parameter.

