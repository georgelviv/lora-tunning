from .utils import (
  parse_msg, format_msg, map_response_to_state, map_config_to_action, estimate_tx_current,
  estimate_tx_energy, estimate_rssi_score, estimate_reward
)

def test_parse_msg():
  assert parse_msg("CONFIG_GET;FQ=868.00,BW=500") == ("CONFIG_GET", [("FQ", 868), ("BW", 500)])
  assert parse_msg("RESET;") == ("RESET", [])
  assert parse_msg("CONFIG_SYNC;BW=500") == ("CONFIG_SYNC", [("BW", 500)])
      
def test_format_msg():
  assert format_msg("CONFIG_GET") == "CONFIG_GET"
  assert format_msg("CONFIG_SYNC", [("BW", 500)]) == "CONFIG_SYNC;BW=500"

def test_map_response_to_state():
  assert map_response_to_state([
    ('ID', 1.0), ('DELAY', 151.0), ('RSSI', -32.0), ('SNR', 7.25), 
    ('TOA', 36.0), ('BPS', 611.0), ('CHC', 1.0), ('ATT', 2)
  ]) ==  {
    'bytes_per_second': 611.0,
    'chunks_count': 1.0,
    'delay': 151.0,
    'rssi': -32.0,
    'snr': 7.25,
    'time_over_air': 36.0,
    'attempt': 2
  }

def test_map_config_to_action():
  assert map_config_to_action([
    ('FQ', 869.0), ('BW', 500.0), ('SF', 8.0), ('CR', 8.0), 
    ('TP', 10.0), ('IH', 0.0), ('HS', 10.0), ('PL', 10.0),
    ('CL', 45.0), ('RT', 1.0)
  ]) ==  {
    'BW': 500.0,
    'CR': 8,
    'CL': 45,
    'FQ': 869.0,
    'HS': 10,
    'IH': False,
    'PL': 10,
    'RT': 1,
    'SF': 8,
    'TP': 10
  }

def test_estimate_tx_current():
  assert estimate_tx_current(20) == 120
  assert estimate_tx_current(15) == 58
  assert estimate_tx_current(10) == 24.5

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
      'delay': 828.0, 'rssi': -11.0, 'snr': 8.0, 'time_over_air': 314.0,
      'bytes_per_second': 1500, 'chunks_count': 1.0, 'attempt': 1
    },
    {
      'FQ': 869.0, 'BW': 500.0, 'SF': 7,
      'CR': 8, 'TP': 10, 'IH': False,
      'HS': 10, 'PL': 10, 'CL': 45, 'RT': 0
    }), 3) == 0.452
  
  assert round(estimate_reward(
    {
      'delay': 828.0, 'rssi': -11.0, 'snr': 8.0, 'time_over_air': 314.0,
      'bytes_per_second': 1500, 'chunks_count': 1.0, 'attempt': 2
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