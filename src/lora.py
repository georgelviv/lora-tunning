from typing import Dict
import serial.tools.list_ports
import logging
import threading
import asyncio
from .utils import format_msg, parse_msg

class Lora:
  def __init__(self, logger: logging.Logger, port_filter: str) -> None:
    self.logger: logging.Logger = logger
    self.ser = None
    self.thread = None
    self.loop = asyncio.get_event_loop()
    self.pending_futures: Dict[str, asyncio.Future] = {}
    self.running = False

    self.serial_port = self.find_serial_port(port_filter)

    if self.serial_port:
      self.start_listener()
    else:
      self.logger.error("Serial port not found")
  
  def find_serial_port(self, filter: str) -> str:
    ports = serial.tools.list_ports.comports()
    for port in ports:
      if port.device.startswith(filter):
        return port.device
    return None
  
  def start_listener(self):
    self.ser = serial.Serial(port=self.serial_port, baudrate=115200, timeout=1)
    self.running = True
    self.thread = threading.Thread(target=self.read_serial, daemon=True)
    self.thread.start()
    self.logger.info(f"Listening thread started on {self.serial_port}")
  
  def read_serial(self):
    self.logger.info(f"Connected to {self.serial_port}")

    try:
      while True:
        if self.ser.in_waiting > 0:
          line = self.ser.readline().decode("utf-8", errors="ignore").strip()
          if line:
            self.serial_handler(line)
    except Exception as e:
      self.logger.error(f"Error in read_serial: {e}")
    finally:
      self.ser.close()
      self.logger.info("Serial connection closed")

  def stop_listener(self):
    self.running = False
    if self.thread and self.thread.is_alive():
      self.thread.join(timeout=2)
    self.logger.info("Listener stopped")

  def write_serial(self, data: str) -> None: 
    if self.ser and self.ser.is_open:
      try:
        self.ser.write((data + "\r\n").encode("utf-8"))
        self.logger.info(f"Sent: {data}")
      except Exception as e:
        self.logger.error(f"Error writing to serial: {e}")
    else:
      self.logger.warning("Serial port is not open")

  async def config_get(self) -> None:
    future = self.loop.create_future()
    self.pending_futures["CONFIG_GET"] = future
    msg = format_msg("CONFIG_GET")
    self.write_serial(msg)
    return await future
  
  async def ping(self, id: int) -> None:
    future = self.loop.create_future()
    self.pending_futures['PING'] = future
    msg = format_msg("PING", [("ID", id)])
    self.write_serial(msg)
    return await future

  def serial_handler(self, msg: str):
    command, params = parse_msg(msg)
    if command:
      match command:
        case "CONFIG_GET":
          future = self.pending_futures.pop("CONFIG_GET", None)
          if future and not future.done():
             self.loop.call_soon_threadsafe(future.set_result, params)
        case "PING_ACK":
          future = self.pending_futures.pop("PING", None)
          if future and not future.done():
             self.loop.call_soon_threadsafe(future.set_result, params)
        case _:
          self.logger.warning(f"Unknown command {command}")

      