import asyncio
from lora_tunning import (
  LoraTunning, getLogger, LoraBase, Algorithm,
  read_args, Args, get_backend, get_alg
)
import logging

async def main() -> None:
  args: Args = read_args()
  logger: logging.Logger = getLogger()
  backend: LoraBase = get_backend(logger, args)
  alg: Algorithm = get_alg(backend, args)

  loraTunning = LoraTunning(logger, backend, alg, args)
  await loraTunning.run()

if __name__ == "__main__":
  asyncio.run(main())
