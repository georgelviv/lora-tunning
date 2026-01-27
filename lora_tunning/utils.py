import argparse
import sys
from .models import Algorithm, ArgAlg, Args, LoraBase, State, Action, ArgEnv
import logging
from lora_hardware_model import LoraHardwareModel
from lora_simulation_model import LoraSimulationModel, EnvironmentModel, AreaType
from .algorithms import (
  MultiArmedBandit, MultiArmedBanditDecay, MultiArmedBanditRewardExponential,
  UCB, GradientBandit
)
from pathlib import Path

def read_args() -> Args:
  parser = argparse.ArgumentParser()

  parser.add_argument("--env", type=str, default='simulation')
  parser.add_argument("--port", type=str, default='/dev/cu.usbserial') 
  parser.add_argument('--alg', type=str, default='mab')
  parser.add_argument('--iterations', type=int, default=1000)
  parser.add_argument('--epsilon', type=float, default=0.9)
  parser.add_argument('--alpha', type=float, default=0.3)
  parser.add_argument('--exploration_factor', type=float, default=0.2)
  parser.add_argument(
    '--has_delays',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Enable delay'
  )

  parser.add_argument('--high_reward', type=float, default=0.35)
  parser.add_argument('--results_dir_name', type=str)

  args = parser.parse_args()

  return {
    'env': ArgEnv(args.env),
    'port': args.port,
    'alg': ArgAlg(args.alg),
    'iterations': args.iterations,
    'epsilon': args.epsilon,
    'alpha': args.alpha,
    'exploration_factor': args.exploration_factor,
    'has_delays': args.has_delays,
    'high_reward': args.high_reward,
    'results_dir_name': args.results_dir_name
  }

def get_backend(logger: logging.Logger, args: Args) -> LoraBase:
  env: ArgEnv = args['env']
  if env == ArgEnv.HARDWARE:
    backend: LoraBase = LoraHardwareModel(logger, args['port'])
  elif env== ArgEnv.SIMULATION:
    env_model: EnvironmentModel = EnvironmentModel(
      name=f"math-100-meters",
      path_loss_exponent=2.5,
      shadow_sigma_db=3.0,
      sigma_noise_db=2.0,
      distance_m=100,
      hb_m = 2,
      hm_m = 5,
      area_type=AreaType.SUBURBAN,
      description=f"Suburban 100 meters"
    )
    backend = LoraSimulationModel(logger, env_model)
  else:
    logger.error(f'Unknown Env {env}')
    sys.exit(1)
  return backend

def get_alg(logger: logging.Logger, backend: LoraBase, args: Args) -> Algorithm:
  alg: ArgAlg = args['alg']
  epsilon: float = args['epsilon']
  alpha: float = args['alpha']
  exploration_factor: float = args['exploration_factor']

  if alg == ArgAlg.mab:
    algorithm: Algorithm = MultiArmedBandit(epsilon=epsilon)
  elif alg == ArgAlg.mab_decay:
    algorithm: Algorithm = MultiArmedBanditDecay(epsilon=epsilon)
  elif alg == ArgAlg.mab_exponential:
    algorithm: Algorithm = MultiArmedBanditRewardExponential(
      epsilon=epsilon, alpha=alpha
    )
  elif alg == ArgAlg.ucb:
    algorithm: Algorithm = UCB(
      epsilon=epsilon, alpha=alpha, exploration_factor=exploration_factor
    )
  elif alg == ArgAlg.gradient:
    algorithm: Algorithm = GradientBandit()
  else:
    logger.error(f'Unknown Alg {alg}')
    sys.exit(1)

  results_dir = get_results_dir(backend, algorithm)
  algorithm.set_results_dir(results_dir)
  return algorithm

def getLogger() -> logging.Logger:
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)
  
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.INFO)
  console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')
  console_handler.setFormatter(console_formatter)

  file_handler = logging.FileHandler('app.log', mode='w')
  file_handler.setLevel(logging.INFO)
  file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
  file_handler.setFormatter(file_formatter)

  logger.addHandler(console_handler)
  logger.addHandler(file_handler)

  return logger

def get_results_dir(backend: LoraBase, algorithm: Algorithm) -> Path:
  return Path(__file__).parent.parent / "results" / backend.name / algorithm.name