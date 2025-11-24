from .utils import (
  estimate_tx_current, estimate_tx_energy, estimate_rssi_score, estimate_reward
)

def test_estimate_tx_current():
  assert estimate_tx_current(20) == 120
  assert estimate_tx_current(15) == 58
  assert estimate_tx_current(10) == 24.5
  assert estimate_tx_current(2) == 15

def test_estimate_tx_energy():
  assert round(estimate_tx_energy(20, 125, 45), 3) == 18.562
  assert round(estimate_tx_energy(20, 250, 45), 3) == 37.125
  assert round(estimate_tx_energy(20, 250, 120), 3) == 99.000
  assert round(estimate_tx_energy(20, 250, 150), 3) == 99.000
  assert round(estimate_tx_energy(8, 50, 45), 3) == 3.547

def test_estimate_rssi_score():
  assert estimate_rssi_score(-50) == 1
  assert estimate_rssi_score(-40) == 1
  assert round(estimate_rssi_score(-80), 3) == 0.694
  assert round(estimate_rssi_score(-120), 3) == 0.286
  assert round(estimate_rssi_score(-130), 3) == 0.184

def test_estimate_reward():
  assert round(estimate_reward(
    {
      'DELAY': 828.0, 'RSSI': -11.0, 'SNR': 8.0, 'TOA': 314.0,
      'BPS': 1500, 'CHC': 1.0, 'ATT': 1
    },
    {
      'FQ': 869.0, 'BW': 500.0, 'SF': 7,
      'CR': 8, 'TP': 10, 'IH': False,
      'HS': 10, 'PL': 10, 'CL': 45, 'RT': 0
    }), 3) == 0.452
  
  assert round(estimate_reward(
    {
      'DELAY': 828.0, 'RSSI': -11.0, 'SNR': 8.0, 'TOA': 314.0,
      'BPS': 1500, 'CHC': 1.0, 'ATT': 2
    },
    {
      'FQ': 869.0, 'BW': 500.0, 'SF': 7,
      'CR': 8, 'TP': 10, 'IH': False,
      'HS': 10, 'PL': 10, 'CL': 45, 'RT': 0
    }), 3) == 0.452
  
  assert round(estimate_reward(
    None,
    {
      'FQ': 869.0, 'BW': 500.0, 'SF': 7,
      'CR': 8, 'TP': 10, 'IH': False,
      'HS': 10, 'PL': 10, 'CL': 45, 'RT': 0
    }), 3) == 0