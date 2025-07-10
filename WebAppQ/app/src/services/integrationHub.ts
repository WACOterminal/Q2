import { authenticatedRequest } from "./auth";

const INTEGRATION_HUB_API_URL = process.env.REACT_APP_INTEGRATION_HUB_API_URL || "/api/integrationhub";

export interface Connector {
    name: string;
    description: string;
    enabled: boolean;
}

export const getConnectors = async (): Promise<Connector[]> => {
    const response = await authenticatedRequest(`${INTEGRATION_HUB_API_URL}/connectors`);
    return response.json();
};

export const enableConnector = async (connectorName: string): Promise<void> => {
    await authenticatedRequest(`${INTEGRATION_HUB_API_URL}/connectors/${connectorName}/enable`, {
        method: "POST",
    });
};

export const disableConnector = async (connectorName: string): Promise<void> => {
    await authenticatedRequest(`${INTEGRATION_HUB_API_URL}/connectors/${connectorName}/disable`, {
        method: "POST",
    });
}; 