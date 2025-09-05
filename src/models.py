from typing import TypedDict

class State(TypedDict):
  delay: int
  rssi: int
  snr: float
  time_over_air: int
  bytes_per_second: int
  chunks_count: int

class Action(TypedDict):
  SF: int
  FQ: int
  BW: int
  CR: int
  TP: int
  IH: int
  HS: int
  PL: int
  CL: int
  RT: int