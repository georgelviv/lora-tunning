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
      "timeOverAir": float(data.get("TOA", 0.0)),
      "bytesPerSecond": float(data.get("BPS", 0.0)),
      "chunksCount": float(data.get("CHC", 0.0))
  }

def map_config_to_action(config: List[Tuple[str, float]]) -> Action:
  data = {k.upper(): v for k, v in config}

  return Action(
    frequency=float(data.get("FW", 0.0)),
    bandwidth=float(data.get("BW", 0.0)),
    spreading_factor=int(data.get("SF", 0.0)),
    coding_rate=int(data.get("CR", 0.0)),
    transmission_power=int(data.get("TP", 0.0)),
    implicit_header=bool(data.get("IH", 0.0)),
    header_size=int(data.get("HS", 0.0)),
    payload_length=int(data.get("PL", 0.0)),
    current_limit=int(data.get("CL", 0.0)),
    retries=int(data.get("RT", 0.0))
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

def estimate_reward(state: State, energy):
  b = norm(state['bytesPerSecond'], 0, 5000)
  e = norm(energy, 0, 5)
  d = norm(state['delay'], 0, 10000)
  s = norm(state['snr'], 0, 20)
  toa = norm(state['timeOverAir'], 0, 10000)

  sensitivity = -148
  rssi_margin = state['rssi'] - sensitivity
  rssi_score = norm(rssi_margin, 0, -60)

  print(f'rssi_score {rssi_score}')

  # rssi_margin = sensitivity - state["rssi"]
  # rssi_pen = clip(rssi_margin / 12, 0.0, 1.0)