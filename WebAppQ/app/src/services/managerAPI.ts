import { w3cwebsocket as W3CWebSocket, IMessageEvent } from "websocket";
import { SearchQuery, SearchResponse } from './types';
import keycloak from "../keycloak";

const API_BASE_URL = process.env.REACT_APP_MANAGERQ_API_URL || 'http://localhost:8000/api/v1';

const getHeaders = () => {
    const headers: { [key: string]: string } = {
        'Content-Type': 'application/json',
    };
    if (keycloak.authenticated && keycloak.token) {
        headers['Authorization'] = `Bearer ${keycloak.token}`;
    }
    return headers;
};

// --- API Calls ---

export const listWorkflows = async (status?: string, skip: number = 0, limit: number = 100) => {
    const params = new URLSearchParams({
        skip: String(skip),
        limit: String(limit),
    });
    if (status) {
        params.append('status', status);
    }
    const response = await fetch(`${API_BASE_URL}/workflows?${params.toString()}`, {
        headers: getHeaders(),
    });
    if (!response.ok) {
        throw new Error('Failed to fetch workflows');
    }
    return response.json();
};

export const listGoals = async () => {
    // This endpoint will be on the IntegrationHub, not the ManagerQ.
    // This is a conceptual change to reflect the new architecture.
    const INTEGRATION_HUB_URL = process.env.REACT_APP_INTEGRATIONHUB_API_URL || 'http://localhost:8002/api/v1';
    
    const response = await fetch(`${INTEGRATION_HUB_URL}/openproject/work-packages`, {
        headers: getHeaders(),
    });
    if (!response.ok) {
        throw new Error('Failed to fetch goals from IntegrationHub');
    }
    return response.json();
};

export const getGoal = async (goalId: string) => {
    const INTEGRATION_HUB_URL = process.env.REACT_APP_INTEGRATIONHUB_API_URL || 'http://localhost:8002/api/v1';
    
    const response = await fetch(`${INTEGRATION_HUB_URL}/openproject/work-packages/${goalId}`, {
        headers: getHeaders(),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch goal details');
    }
    return response.json();
};

export const createGoal = async (goal: { subject: string, project_id: number }) => {
    const INTEGRATION_HUB_URL = process.env.REACT_APP_INTEGRATIONHUB_API_URL || 'http://localhost:8002/api/v1';
    
    const response = await fetch(`${INTEGRATION_HUB_URL}/openproject/work-packages`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify(goal),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to create goal');
    }
    return response.json();
};

export const getWorkflow = async (workflowId: string) => {
    const response = await fetch(`${API_BASE_URL}/workflows/${workflowId}`, {
        headers: getHeaders(),
    });
    if (!response.ok) {
        throw new Error('Failed to fetch workflow details');
    }
    return response.json();
};

export const createWorkflow = async (prompt: string, context?: object) => {
    const response = await fetch(`${API_BASE_URL}/workflows`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({ prompt, context }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to create workflow');
    }
    return response.json();
};

export const approveWorkflowTask = async (workflowId: string, taskId: string, approved: boolean) => {
    const response = await fetch(`${API_BASE_URL}/workflows/${workflowId}/tasks/${taskId}/approve`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({ approved }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to approve task');
    }
    // This endpoint returns 204 No Content, so we don't return JSON
};

export const cognitiveSearch = async (searchQuery: SearchQuery): Promise<SearchResponse> => {
    const response = await fetch(`${API_BASE_URL}/search/`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify(searchQuery),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to perform search');
    }
    return response.json();
};

export const respondToApproval = async (workflowId: string, taskId: string, approved: boolean): Promise<void> => {
    const response = await fetch(`${API_BASE_URL}/workflows/${workflowId}/tasks/${taskId}/respond`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({ approved }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to respond to approval');
    }
};

export const knowledgeGraphQuery = async (query: string): Promise<any> => {
    const response = await fetch(`${API_BASE_URL}/search/kg-query`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({ query }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to perform knowledge graph query');
    }
    return response.json();
};

export const getNodeNeighbors = async (nodeId: string): Promise<any> => {
    const response = await fetch(`${API_BASE_URL}/search/kg-neighbors`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({ node_id: nodeId }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch node neighbors');
    }
    return response.json();
};

export const controlWorkflow = async (workflowId: string, action: 'pause' | 'resume' | 'cancel') => {
    const response = await fetch(`${API_BASE_URL}/workflows/${workflowId}/control`, {
        method: 'POST',
        headers: getHeaders(),
        body: JSON.stringify({ action }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `Failed to ${action} workflow`);
    }
    return response.json();
};

export const ingestDataSource = async (data: { url: string }): Promise<any> => {
    const token = keycloak.token; // Assuming keycloak.token is available here
    if (!token) {
        throw new Error("Authentication token not found.");
    }

    const response = await fetch(`${API_BASE_URL}/v1/ingestion/web`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(data)
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to ingest data source' }));
        throw new Error(errorData.detail);
    }

    return response.json();
};

export const generateWorkflowFromPrompt = async (description: string): Promise<{ task_id: string }> => {
    const token = keycloak.token; // Assuming keycloak.token is available here
    if (!token) {
        throw new Error("Authentication token not found.");
    }

    const response = await fetch(`${API_BASE_URL}/v1/workflows/generate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ description })
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to start workflow generation.' }));
        throw new Error(errorData.detail);
    }

    return response.json();
};

// --- NEW: Report fetching functions ---

export const getFinOpsReport = async (): Promise<any> => {
    const token = keycloak.token; // Assuming keycloak.token is available here
    if (!token) {
        throw new Error("Authentication token not found.");
    }
    const response = await fetch(`${API_BASE_URL}/v1/reports/finops`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!response.ok) throw new Error('Failed to fetch FinOps report');
    return response.json();
};

export const getSecurityReport = async (): Promise<any> => {
    const token = keycloak.token; // Assuming keycloak.token is available here
    if (!token) {
        throw new Error("Authentication token not found.");
    }
    const response = await fetch(`${API_BASE_URL}/v1/reports/security`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!response.ok) throw new Error('Failed to fetch Security report');
    return response.json();
};

export const getRCAReports = async (): Promise<any[]> => {
    const token = keycloak.token; // Assuming keycloak.token is available here
    if (!token) {
        throw new Error("Authentication token not found.");
    }
    const response = await fetch(`${API_BASE_URL}/v1/reports/rca`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!response.ok) throw new Error('Failed to fetch RCA reports');
    return response.json();
};

export const getStrategicBriefing = async (): Promise<any> => {
    const token = keycloak.token; // Assuming keycloak.token is available here
    if (!token) {
        throw new Error("Authentication token not found.");
    }
    const response = await fetch(`${API_BASE_URL}/v1/reports/strategic-briefing`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!response.ok) throw new Error('Failed to fetch Strategic Briefing');
    return response.json();
};

export const getPredictiveScalingReport = async (): Promise<any[]> => {
    const token = keycloak.token; // Assuming keycloak.token is available here
    if (!token) {
        throw new Error("Authentication token not found.");
    }
    const response = await fetch(`${API_BASE_URL}/v1/reports/predictive-scaling`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    if (!response.ok) throw new Error('Failed to fetch Predictive Scaling report');
    return response.json();
};

// --- WebSocket Management ---

let dashboardSocketClient: W3CWebSocket | null = null;
let logSocketClient: W3CWebSocket | null = null;

export const connectToDashboardSocket = (onMessageCallback: (message: any) => void) => {
    if (dashboardSocketClient && dashboardSocketClient.readyState === dashboardSocketClient.OPEN) {
        console.log('WebSocket is already connected.');
        return;
    }

    const wsUrl = (API_BASE_URL.replace('http', 'ws')).split('/api/v1')[0] + '/api/v1/dashboard/ws';
    
    dashboardSocketClient = new W3CWebSocket(wsUrl);

    dashboardSocketClient.onopen = () => {
        console.log('WebSocket Client Connected');
    };

    dashboardSocketClient.onmessage = (message: IMessageEvent) => {
        try {
            const data = JSON.parse(message.data as string);
            onMessageCallback(data);
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };

    dashboardSocketClient.onerror = (error: Error) => {
        console.error('WebSocket Error:', error);
    };

    dashboardSocketClient.onclose = () => {
        console.log('WebSocket Client Closed');
        // Optional: Implement automatic reconnection logic here
    };
};

export const disconnectFromDashboardSocket = () => {
    if (dashboardSocketClient) {
        dashboardSocketClient.close();
        dashboardSocketClient = null;
    }
};

export const connectToTaskLogsSocket = (taskId: string, onMessageCallback: (message: any) => void) => {
    if (logSocketClient) {
        logSocketClient.close(); // Close previous connection if any
    }

    const wsUrl = (API_BASE_URL.replace('http', 'ws')).split('/api/v1')[0] + `/api/v1/logs/ws/${taskId}`;
    
    logSocketClient = new W3CWebSocket(wsUrl);

    logSocketClient.onopen = () => console.log(`Log WebSocket connected for task ${taskId}`);
    logSocketClient.onmessage = (message: IMessageEvent) => {
        try {
            const data = JSON.parse(message.data as string);
            onMessageCallback(data);
        } catch (error) {
            console.error('Error parsing log message:', error);
        }
    };
    logSocketClient.onerror = (error: Error) => console.error(`Log WebSocket error for task ${taskId}:`, error);
    logSocketClient.onclose = () => console.log(`Log WebSocket closed for task ${taskId}`);
};

export const disconnectFromLogsSocket = () => {
    if (logSocketClient) {
        logSocketClient.close();
        logSocketClient = null;
    }
}; 