# RL Project

## Table of Contents
1. [Overview](#overview)
2. [Development](#development)
3. [Reproducibility](#reproducibility)
4. [Project Structure](#project-structure)
5. [Reusing the package](#reusing-the-package)
5. [Contributors](#contributors)

## Overview
This repo contains the code for a Reinforcement Learning (RL) project in which the goal is to implement different RL algorithms to solve a specific environment ([highway-env](https://highway-env.farama.org/))

## Development
This project follows the best practices we currently rely on for building maintainable Python projects. We use:

- **`uv`** for dependency management  
- **`pre-commit`** for automated code quality checks  
- **`Ruff`** for linting and formatting (a VS Code configuration is included)  
- **`Makefile`** to run common commands consistently

### Requirements
- **Python 3.12**
- [**`uv`**](https://docs.astral.sh/uv/) (recommended) for dependency management and editable installs (alternatively, you can use `pip install -e .` and manage dependencies via `requirements.txt`).
- **`make`** (recommended) to use the provided Makefile commands (alternatively, you can execute the underlying commands manually).

### Environment setup
Install dependencies and set up `pre-commit` hooks:
```bash
make install
```

### Code quality
Run linting/formatting checks via `pre-commit`:
```bash
make pre-commit
```

## Reproducibility

### Training
To reproduce the project results you can run the training scripts for the type of model [`(ModelType)`](./src/rl_project/models/core/typing.py) you want to train. To do this you can run the following command:
```bash
make training SEED=YourSeed MODEL=YourModelType ENV_TYPE=YourEnvType N_STEPS=NumberOfSteps
```
For more information on all the parameters possible, you can see the help of the command with:
```bash
make training HELP=1
```
**Note**: To use an option use the full name of the option in capital letters, for example : --n_steps $\rightarrow$ N_STEPS


### Evaluation
Finally, to evaluate a model on a fixed number of episodes you can run the following command:
```bash
make evaluation MODEL_PATH=YourModelPath SEED=YourSeed N=NumberOfEpisodes
```
For more information on all the parameters possible, you can see the help of the command with:
```bash
make evaluation HELP=1
```
**Note**: To use an option use the full name of the option in capital letters, for example : --model-path $\rightarrow$ MODEL_PATH

## Project Structure
This repository is organized like a typical Python package so that you can reuse the structure easily for future projects. You will then find 
- in the [`rl_project`](src/rl_project/) folder the core package
- in the [`scripts`](src/scripts/) folder the scripts to run the training and evaluation of the models, as well as the visualization of the results

The rest of the folders are organized as follows:

- **[`data/`](./data/)** : Folder to store all the models and results of the project

## Reusing the package
You can also install this package independently in any of your projects with:
```bash
uv add git+https://github.com/Bliights/RL-project
```
or
```bash
pip install git+https://github.com/Bliights/RL-project
```


## Contributors

|            Name            |                   Email                   |
| :------------------------: | :---------------------------------------: |
|    MOLLY-MITTON Clément    |    clement.molly-mitton@student-cs.fr     |
|       VERBECQ DIANE        |        diane.verbecq@student-cs.fr        |
|       BADOULES Fanny       |       fanny.badoules@student-cs.fr        |
|       KEMICHE Ghiles       |       ghiles.kemiche@student-cs.fr        |