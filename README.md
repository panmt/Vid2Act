# Model-Based Reinforcement Learning with Multi-Task Offline Pretraining (ECML 2024)


#### Model-Based Reinforcement Learning with Multi-Task Offline Pretraining [[arXiv]](https://arxiv.org/pdf/2306.03360)

Minting Pan*, Yitao Zheng*, Yunbo Wang, Xiaokang Yang

## Setting up

### Create an environment

```
conda env create -f env.yaml
```

## Experiments

### Fine-tuning command on Meta-World:

```
python dreamer.py \
--logdir \
path/to/log \
--config \
defaults metaworld \
--task \
target_metaworld_task \
--video_dir \
path/to/offline/datasets \
--pretrain_model_dir \
path/to/teacher/model \
--source_tasks \
['list', 'of', 'source', 'metaworld', 'tasks']
```

### Fine-tuning command on Deepmind Control Suite:

```
python dreamer.py \
--logdir \
path/to/log \
--config \
defaults dmc \
--task \
target_dmc_task \
--video_dir \
path/to/offline/datasets \
--pretrain_model_dir \
path/to/teacher/model \
--source_tasks \
['list', 'of', 'source', 'dmc', 'tasks']
```

## Acknowledgement
We appreciate the following github repos where we borrow code from:

https://github.com/jsikyoon/dreamer-torch
