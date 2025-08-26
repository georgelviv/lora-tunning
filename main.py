import serial.tools.list_ports
from dotenv import load_dotenv
import os

load_dotenv(override=True)

PORT_FILTER = os.getenv('PORT_FILTER')

def find_serial_port(filter: str) -> str:
  ports = serial.tools.list_ports.comports()
  for port in ports:
    if port.device.startswith(filter):
      return port.device
  return None

def read_serial(port: str):
  ser = serial.Serial(
    port="/dev/ttyUSB0",  # або "COM3" на Windows
      baudrate=9600,
      timeout=1  # щоб не блокувався назавжди
  )

serial_port = find_serial_port(PORT_FILTER)
print(serial_port)


  