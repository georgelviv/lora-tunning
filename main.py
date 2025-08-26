from dotenv import load_dotenv
import os
from lora_tunning import LoraTunning

load_dotenv(override=True)

PORT_FILTER = os.getenv('PORT_FILTER')

def main() -> None:
  LoraTunning(PORT_FILTER)

if __name__ == "__main__":
    main()


  