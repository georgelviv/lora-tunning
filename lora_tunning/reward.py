from .constants import RSSI_MAX, SX1276_SENSITIVITY
from .models import State

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

def estimate_reward(state: State):
  if not state:
    return 0

  bps = norm(state['BPS'], 0, 3000)
  energy = norm(state['ETX'], 0, 100)
  delay = norm(state['DELAY'], 0, 60000)
  snr = norm(state['SNR'], 0, 20)
  toa = norm(state['TOA'], 0, 10000)

  rssi_score = estimate_rssi_score(state['RSSI'])

  reward = (
    0.4 * bps
    + 0.2 * snr
    + 0.2 * rssi_score
    - 0.1 * energy
    - 0.05 * delay
    - 0.05 * toa
  )

  return round(reward, 10)
  