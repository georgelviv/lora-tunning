from lora_tunning import estimate_tx_energy

def test_addition():
  estimate_tx_energy(13, 5, 100)
  assert estimate_tx_energy(13, 45, 100) == 4.31