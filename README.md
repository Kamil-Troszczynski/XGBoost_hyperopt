## Autor projektu: Kamil Troszczyński


### Environment setup

#### Firstly, open terminal and paste this command:

```bash 
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Then, you can install all needed packages which are required for a project:

```bash 
uv sync
```

#### Now, you can activate .venv environment:
```bash
source .venv/bin/activate
```
Then go to [Constraints.py](Optimization/Constraints.py) file in order to change limits of search space.
In [PreparedData.py](Optimization/PreparedData.py), you can change booster configuration, proportion for validation dataset and number of trials.
```bash
###   After these steps, you can run optimizers by commands:

optim_name = {CMA, DE, OnePlusOne, Optuna, PSO, RS, TBPSA}

uv run Optimization/<optim_name>.py

###   Or with usual interpreter:

python Optimization/<optim_name>.py
```

#### If you want to deactivate .venv environment:
```bash
deactivate
```

### Project results are described here:
[XGBoost hyperparameters optimization results with different optimizers](POP_raport.pdf)