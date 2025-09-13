from typing import List, Tuple
from .models import State, Action
from .constants import RSSI_MAX, SX1276_SENSITIVITY, SX1276_TX_CURRENT

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

def map_response_to_state(response: List[Tuple[str, float]]) -> State:
  data = {k.upper(): v for k, v in response}

  return {
    "delay": float(data.get("DELAY", 0.0)),
    "rssi": float(data.get("RSSI", 0.0)),
    "snr": float(data.get("SNR", 0.0)),
    "time_over_air": float(data.get("TOA", 0.0)),
    "bytes_per_second": float(data.get("BPS", 0.0)),
    "chunks_count": float(data.get("CHC", 0.0)),
    "attempt": int(data.get("ATT", 0.0))
  }

def map_config_to_action(config: List[Tuple[str, float]]) -> Action:
  data = {k.upper(): v for k, v in config}

  return Action(
    FQ=int(data.get("FQ", 0.0)),
    BW=int(data.get("BW", 0.0)),
    SF=int(data.get("SF", 0.0)),
    CR=int(data.get("CR", 0.0)),
    TP=int(data.get("TP", 0.0)),
    IH=int(data.get("IH", 0.0)),
    HS=int(data.get("HS", 0.0)),
    PL=int(data.get("PL", 0.0)),
    CL=int(data.get("CL", 0.0)),
    RT=int(data.get("RT", 0.0))
  )

def estimate_tx_current(tx_power: int) -> float:
  powers = sorted(SX1276_TX_CURRENT.keys())
  if tx_power in SX1276_TX_CURRENT:
    return SX1276_TX_CURRENT[tx_power]
  
  for i in range(len(powers) - 1):
    low_p = powers[i]
    high_p = powers[i + 1]
    if low_p < tx_power < high_p:
      low_i = SX1276_TX_CURRENT[low_p]
      high_i = SX1276_TX_CURRENT[high_p]
      interp_i = low_i + (high_i - low_i) * (tx_power - low_p) / (high_p - low_p)
      return interp_i

def estimate_tx_energy(tx_power_dbm: float, toa_s: float, current_limit: int, voltage_v: float = 3.3) -> float:
  current_mA = estimate_tx_current(tx_power_dbm)
  current_mA = min(current_limit, current_mA)
  current_a = current_mA / 1000.0
  energy_j = voltage_v * current_a * toa_s

  return energy_j

def clip(x, lo=0.0, hi=1.0) -> float:
  return max(lo, min(hi, x))

def norm(x, xmin, xmax) -> float:
  if xmax == xmin:
    return 0.0
  return clip((x - xmin) / (xmax - xmin), 0.0, 1.0)

def estimate_rssi_score(rssi: int) -> float:
  rssi_score = (rssi - SX1276_SENSITIVITY) / (RSSI_MAX - SX1276_SENSITIVITY)
  rssi_score = max(0.0, min(1.0, rssi_score))

  return rssi_score

def estimate_reward(state: State, action: Action):
  if not state:
    return 0

  energy = estimate_tx_energy(
    action['TP'], state['time_over_air'],
    action['CL']
  )

  b = norm(state['bytes_per_second'], 0, 3000)
  e = norm(energy, 0, 100)
  d = norm(state['delay'], 0, 60000)
  s = norm(state['snr'], 0, 20)
  toa = norm(state['time_over_air'], 0, 10000)

  rssi_score = estimate_rssi_score(state['rssi'])

  print(b, s, rssi_score, e, d, toa)

  reward = (
    0.4 * b
    + 0.2 * s
    + 0.2 * rssi_score
    - 0.1 * e
    - 0.05 * d
    - 0.05 * toa
  )

  return round(reward, 10)

def current_limit_for_tp(tp: int) -> int:
  if tp <= 7:
    return 80
  elif tp <= 13:
    return 100
  elif tp <= 17:
    return 120
  else:
    return 140