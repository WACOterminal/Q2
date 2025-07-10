class BaseConnector:
    """
    The base class for all IntegrationHub connectors.
    """
    def __init__(self, config):
        self.config = config

    def get_name(self) -> str:
        """
        Returns the name of the connector.
        """
        return self.__class__.__name__ 