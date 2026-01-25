# Lora Tunning

`poetry run python main.py --env simulation --alg mab --iteration 1`

## Args
- `--port` - port filtering string. Valid for `hardware` only. Default `/dev/cu.usbserial`
- `--env` - environment. Options `hardware`, `simulation`. Default `simulation`
- `--alg` - algorithm. Options `mab`. Default `mab`
- `--iterations` - iterations count. Default `1000`

## Tests

`poetry run pytest`