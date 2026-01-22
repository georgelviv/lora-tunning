from abc import ABC, abstractmethod
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

class LoraBase(ABC):
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