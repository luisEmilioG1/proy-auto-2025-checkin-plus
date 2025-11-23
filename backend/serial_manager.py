import serial
import serial.tools.list_ports
import time
import logging
from typing import Optional
from config import SERIAL_BAUD_RATE, SERIAL_PORT

logger = logging.getLogger(__name__)


class SerialManager:
    def __init__(self, baud_rate: int = SERIAL_BAUD_RATE, port: Optional[str] = SERIAL_PORT):
        self.baud_rate = baud_rate
        self.port = port
        self.serial_connection: Optional[serial.Serial] = None
        self.is_connected_flag = False
    
    def find_arduino_port(self) -> Optional[str]:
        """Find Arduino serial port automatically"""
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            port_description = port.description.upper()
            if any(keyword in port_description for keyword in ['ARDUINO', 'CH340', 'CH341', 'USB']):
                logger.info(f"Found potential Arduino port: {port.device}")
                return port.device
        
        if ports:
            logger.info(f"Using first available port: {ports[0].device}")
            return ports[0].device
        
        return None
    
    def connect(self) -> bool:
        """Connect to Arduino via serial"""
        if self.serial_connection and self.serial_connection.is_open:
            return True
        
        port = self.port or self.find_arduino_port()
        
        if port is None:
            logger.error("No Arduino port found")
            return False
        
        try:
            self.serial_connection = serial.Serial(port, self.baud_rate, timeout=1)
            time.sleep(2)
            self.is_connected_flag = True
            logger.info(f"Connected to Arduino on {port}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Arduino: {e}")
            self.is_connected_flag = False
            return False
    
    def send_command(self, command: str) -> bool:
        """Send command to Arduino"""
        if not self.is_connected():
            if not self.connect():
                return False
        
        if not self.serial_connection or not self.serial_connection.is_open:
            return False
        
        try:
            message = f"{command}\n"
            self.serial_connection.write(message.encode('utf-8'))
            logger.debug(f"Sent to Arduino: {command}")
            return True
        except Exception as e:
            logger.error(f"Error sending to Arduino: {e}")
            self.is_connected_flag = False
            return False
    
    def is_connected(self) -> bool:
        """Check if Arduino is connected"""
        if self.serial_connection and self.serial_connection.is_open:
            return self.is_connected_flag
        return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.is_connected_flag = False
            logger.info("Disconnected from Arduino")

