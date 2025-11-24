from typing import TypedDict

class State(TypedDict):
  DELAY: float
  RSSI: float
  SNR: float
  TOA: float
  BPS: float
  CHC: float
  ATT: float

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