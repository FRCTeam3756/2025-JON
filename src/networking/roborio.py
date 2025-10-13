import os
import json
import logging
from logs.logging_setup import setup_logger
from typing import Optional, Dict, Any
from config import NetworkingConfig

from networktables import NetworkTables

##############################################################################

class RoboRio:
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

        NetworkTables.initialize(server=NetworkingConfig.ROBOT_IP_ADDRESS)
        self.table = NetworkTables.getTable(NetworkingConfig.NETWORK_TABLE_NAME)
        self.logger.info(
            f'NetworkTables initialized with server: {NetworkingConfig.ROBOT_IP_ADDRESS}')

    def send_data(self, data: Dict[str, Any]) -> None:
        """Posts JSON data to NetworkTables."""
        if not self.network_tables_connection():
            self.logger.warning(
                "NetworkTables is not connected. Data not sent.")
            return

        try:
            json_data = json.dumps(data)
            self.table.putString(NetworkingConfig.DATA_ENTRY_NAME, json_data)
            self.logger.info(
                f'Data successfully posted to NetworkTables: {json_data}')
        except (TypeError, ValueError) as e:
            self.logger.error(f'Failed to serialize data to JSON: {e}')

    def get_data(self, data) -> Optional[str]:
        """Pulls JSON data from NetworkTables and returns it as a dictionary."""
        if not self.network_tables_connection():
            self.logger.warning(
                "NetworkTables is not connected. Returning None.")
            return None

        json_data = self.table.getString(data, "{}")
        try:
            return json.loads(json_data)
        except json.JSONDecodeError as e:
            self.logger.error(f'Error decoding JSON from NetworkTables: {e}')
            return None

    @staticmethod
    def network_tables_connection() -> bool:
        """Checks if NetworkTables is connected to the robot."""
        connected = NetworkTables.isConnected()
        return connected
