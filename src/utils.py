from .models import State, Action
import logging
from .constants import RSSI_MAX, SX1276_SENSITIVITY, SX1276_TX_CURRENT


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
    action['TP'], state['TOA'],
    action['CL']
  )

  b = norm(state['BPS'], 0, 3000)
  e = norm(energy, 0, 100)
  d = norm(state['DELAY'], 0, 60000)
  s = norm(state['SNR'], 0, 20)
  toa = norm(state['TOA'], 0, 10000)

  rssi_score = estimate_rssi_score(state['RSSI'])

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
  
def getLogger() -> logging.Logger:
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)
  
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)
  console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')
  console_handler.setFormatter(console_formatter)

  file_handler = logging.FileHandler('app.log', mode='w')
  file_handler.setLevel(logging.INFO)
  file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
  file_handler.setFormatter(file_formatter)

  logger.addHandler(console_handler)
  logger.addHandler(file_handler)

  return logger