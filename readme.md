# Lora Tunning

```bash
poetry run python main.py --env simulation --alg mab --iteration 1
```

```bash
poetry run python main.py --env hardware --alg mab_decay
```

## Main Arguments
- `--port` - port filtering string. Valid for `hardware` only. Default `/dev/cu.usbserial`
- `--env` - environment. Options `hardware`, `simulation`. Default `simulation`
- `--alg` - algorithm. More in algorithms sections Options `mab` Default `mab`
- `--iterations` - iterations count. Default `1000`
- `--has_delays` - has delays. Default `False`
- `--high_reward` - definition of high reward. Default `0.35`
- `--results_dir_name` - results folder name. If not provided, name of algorithm will be used.

## Algorithms

### MAB
Multi Armed Bandit. To use this algorithm, run:
`poetry run python main.py --alg mab`

Configurations:
`--epsilon` - epsilon value. Default `0.9`

### MAB Decay
Multi Armed Bandit with Decay. To use this algorithm, run:
`poetry run python main.py --alg mab_decay`

Configurations:
`--epsilon` - epsilon value. Default `0.9`

### MAB Exponential
Multi Armed Bandit with exponential reward. To use this algorithm, run:
`poetry run python main.py --alg mab_exponential`

Configurations:
`--epsilon` - epsilon value. Default `0.9`
`--alpha` - decay with every iteration value. Default `0.3`

### UCB
Upper Confidence Bound. To use this algorithm, run:
`poetry run python main.py --alg ucb`

Configurations:
`--epsilon` - epsilon for secondary action. Default `0.9`
`--alpha` - decay with every iteration value. Default `0.3`
`--exploration_factor` - exploration factor value. Default `0.2`

### Gradient
Gradient. To use this algorithm, run:
`poetry run python main.py --alg gradient`

Configurations:
`--epsilon` - epsilon value. Default `0.9`


## Tests

```bash
poetry run pytest
```