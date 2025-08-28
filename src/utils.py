from typing import List, Tuple


def parse_msg(msg: str) -> Tuple[str, List[Tuple[str, float]]]:
  command = None
  params = []
  if ";" in msg:
    command, params_str = msg.split(";", 1)
    params_pairs = []
    if "," in params_str:
      params_pairs = params_str.split(",")
    elif params_str:
      params_pairs = [params_str]

    for param_pair in params_pairs:
      key, val = param_pair.split("=", 1)

      try:
        val_conv = float(val)
      except ValueError:
        val_conv = val

      params.append((key, val_conv))

  return (command, params)

def format_msg(command: str, params: List[Tuple[str, float]] = []):
  if not params:
    return command
  params_str = ",".join(f"{key}={value}" for key, value in params)
  return f"{command};{params_str}"
