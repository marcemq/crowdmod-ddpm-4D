# Import the W&B Python Library and log into W&B
import functools
import wandb

# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score

def main(myVar1, myVar2):
    print(f'parametro pasado:{myVar1}')
    print(f'otro:{myVar2}')
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})

# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
var1=5
var2="hola"
wandb.agent(sweep_id, function=functools.partial(main, var1, var2), count=3)