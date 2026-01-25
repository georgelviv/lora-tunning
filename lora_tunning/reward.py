from .constants import RSSI_MAX, SX1276_SENSITIVITY, SX1276_TX_CURRENT
from .models import State, Action

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

def estimate_reward(state: State, action: Action):
  if not state:
    return 0

  energy = estimate_tx_energy(
    action['TP'], state['TOA'],
    action['CL']
  )

  bps = norm(state['BPS'], 0, 3000)
  energy_norm = norm(energy, 0, 100)
  delay = norm(state['DELAY'], 0, 60000)
  snr = norm(state['SNR'], 0, 20)
  toa = norm(state['TOA'], 0, 10000)

  rssi_score = estimate_rssi_score(state['RSSI'])

  reward = (
    0.4 * bps
    + 0.2 * snr
    + 0.2 * rssi_score
    - 0.1 * energy_norm
    - 0.05 * delay
    - 0.05 * toa
  )

  print('----REWARD START----')
  print('BPS', bps)
  print('SNR', snr)
  print('RSSI', rssi_score)
  print('ENERGY', energy)
  print('ETX', state['ETX'])

  print('----REWARD END----')

  return round(reward, 10)
  