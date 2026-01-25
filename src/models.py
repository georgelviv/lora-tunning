from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict
from enum import StrEnum

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

class ArgEnv(StrEnum):
  SIMULATION = 'simulation'
  HARDWARE = 'hardware'

class ArgAlg(StrEnum):
  mab = 'mab'

class Args(TypedDict):
  env: ArgEnv
  port: str
  alg: ArgAlg
  iterations: int

class LoraBase(ABC):
  @property
  @abstractmethod
  def name(self) -> str:
      pass

  @abstractmethod
  async def start(self):
    pass

  @abstractmethod
  async def stop(self):
    pass

  @abstractmethod
  async def config_get(self) -> Action:
    pass

  @abstractmethod
  async def ping(self, id: int) -> State:
    pass

  @abstractmethod
  async def config_sync(self, id: int, params) -> None:
    pass


class Algorithm(ABC):
  @property
  @abstractmethod
  def name(self) -> str:
      pass
  
  @abstractmethod
  def set_results_dir(results_dir: Path) -> None:
    pass

  @abstractmethod
  def choose_action(self) -> Action:
    pass

  @abstractmethod
  def update(action: Action, reward: float) -> None:
    pass

  @abstractmethod
  def save() -> None:
    pass

  @abstractmethod
  def get_iteration() -> int:
    pass