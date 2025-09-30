from typing import TypedDict

class PrimaryAction(TypedDict):
  SF: int
  BW: int
  CR: int
  IH: int

class SecondaryAction(TypedDict):
  FQ: int
  TP: int
  IH: int
  HS: int
  PL: int
  CL: int
  RT: int