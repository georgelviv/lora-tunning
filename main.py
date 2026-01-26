import asyncio
from lora_tunning import (
  LoraTunning, getLogger, LoraBase, Algorithm,
  read_args, Args, get_backend, get_alg, analyse
)
import logging

async def main() -> None:
  args: Args = read_args()
  logger: logging.Logger = getLogger()
  backend: LoraBase = get_backend(logger, args)
  algorithm: Algorithm = get_alg(logger, backend, args)

  loraTunning = LoraTunning(logger, backend, algorithm, args)
  await loraTunning.run()
  analyse(logger, backend, algorithm)

if __name__ == "__main__":
  asyncio.run(main())
