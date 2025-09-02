from typing import TypedDict

class State(TypedDict):
  delay: float
  rssi: float
  snr: float
  timeOverAir: float
  bytesPerSecond: float
  chunksCount: float

class Action(TypedDict):
  frequency: float
  bandwidth: float
  spreading_factor: int
  coding_rate: int
  transmission_power: int
  implicit_header: bool
  header_size: int
  payload_length: int
  current_limit: int
  retries: int