from .utils import parse_msg

def test_parse_msg():
  assert parse_msg("CONFIG_GET;FW=868.00,BW=500") == ("CONFIG_GET", [("FW", 868), ("BW", 500)])
  assert parse_msg("RESET;") == ("RESET", [])
  assert parse_msg("CONFIG_SYNC;BW=500") == ("CONFIG_SYNC", [("BW", 500)])
      
