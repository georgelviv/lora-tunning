import asyncio
from dotenv import load_dotenv
import os
from src import (
  LoraTunning, getLogger, LoraBase, Algorithm, MultiArmedBandit, MultiArmedBanditDecay
)
import logging
from pathlib import Path
from lora_hardware_model import LoraHardwareModel
from lora_simulation_model import LoraSimulationModel, EnvironmentModel, AreaType

load_dotenv(override=True)

PORT_FILTER = os.getenv('PORT_FILTER')

def get_alg(logger: logging.Logger) -> Algorithm:
  alg_name = "mab"
  results_dir = Path(__file__).parent / "results" / alg_name
  algorithm: Algorithm = MultiArmedBandit(results_dir)
  # algorithm: Algorithm = MultiArmedBanditDecay()
  return algorithm

def get_backend(logger: logging.Logger) -> LoraBase:
  # backend: LoraBase = LoraHardwareModel(logger, PORT_FILTER)
  env_model: EnvironmentModel = EnvironmentModel(
    name=f"math-100-meters",
    path_loss_exponent=2.5,
    shadow_sigma_db=3.0,
    sigma_noise_db=2.0,
    distance_m=100,
    hb_m = 1.2,
    hm_m = 1.0,
    area_type=AreaType.SUBURBAN,
    description=f"Suburban 100 meters"
  )
  backend = LoraHardwareModel(logger, env_model)
  return backend


async def main():
  logger: logging.Logger = getLogger()
  alg: Algorithm = get_alg(logger)
  backend: LoraBase = get_backend(logger)

  loraTunning = LoraTunning(logger, backend, alg)
  await loraTunning.run()

if __name__ == "__main__":
  asyncio.run(main())
