# example-wandb-trainer

This repository is an example of training repository which integrate with WandB for login.
Please install [poetry](https://python-poetry.org/docs/) to install all the dependencies

How to use :

1. Install dependencies

```shell
poetry install
```

2. Login to wandb

```
poetry run wandb login
```

3. Run the script

```shell
poetry run python example_wandb_trainer/main.py --help
Usage: main.py [OPTIONS]

Options:
  --dataset-path TEXT             Path to iris csv dataset  [default:
                                  ../example-dvc-dataset/dataset/iris.csv]
  --wandb-entity TEXT             WandB user profile  [required]
  --experiment-type [svc|decision_tree]
                                  Experiment model type  [required]
```

E.g.

```shell
poetry run python example_wandb_trainer/main.py --wandb-entity alvin-prayuda --experiment-type svc
```

Example output:

```shell
wandb: Synced svc: https://wandb.ai/alvin-prayuda/experiment-tracking-example/runs/173aitq9
wandb: Synced 5 W&B file(s), 5 media file(s), 5 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20220322_165209-173aitq9/logs
```

4. The dumped model will be available at `./tmp` directory on your local working directory
