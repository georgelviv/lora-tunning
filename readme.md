# Lora Tunning

```bash
poetry run python main.py --env simulation --alg mab --iteration 1
```

```bash
poetry run python main.py --env hardware --alg mab_decay
```

## Args
- `--port` - port filtering string. Valid for `hardware` only. Default `/dev/cu.usbserial`
- `--env` - environment. Options `hardware`, `simulation`. Default `simulation`
- `--alg` - algorithm. Options `mab`, `mab_decay`, `mab_exponential`, `ucb`, `gradient`. Default `mab`
- `--iterations` - iterations count. Default `1000`

## Tests

```bash
poetry run pytest
```