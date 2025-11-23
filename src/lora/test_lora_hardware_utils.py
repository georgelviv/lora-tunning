from .lora_hardware_utils import format_msg, parse_msg, map_config_to_action, map_response_to_state

def test_format_msg():
  assert format_msg("CONFIG_GET") == "CONFIG_GET"
  assert format_msg("CONFIG_SYNC", [("BW", 500)]) == "CONFIG_SYNC;BW=500"

def test_parse_msg():
  assert parse_msg("CONFIG_GET;FQ=868.00,BW=500") == ("CONFIG_GET", [("FQ", 868), ("BW", 500)])
  assert parse_msg("RESET;") == ("RESET", [])
  assert parse_msg("CONFIG_SYNC;BW=500") == ("CONFIG_SYNC", [("BW", 500)])
  
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