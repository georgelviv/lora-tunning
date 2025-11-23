import logging
from .lora_base import LoraBase

class LoraSimulation(LoraBase):
  def __init__(self, logger: logging.Logger):
    self.logger = logger