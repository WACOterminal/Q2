// WebAppQ/app/src/services/workflowBuilderAPI.ts
import { Workflow } from './types';

const API_BASE_URL = 'http://localhost:8000/api/v1'; // Replace with your actual API base URL

export const saveWorkflow = async (workflow: Partial<Workflow>): Promise<Workflow> => {
  const response = await fetch(`${API_BASE_URL}/workflows`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(workflow),
  });

  if (!response.ok) {
    throw new Error('Failed to save workflow');
  }

  return response.json();
};

export const getAgentPersonalities = async (): Promise<string[]> => {
    // In a real application, this would fetch the agent personalities from the managerQ service.
    // For now, we will return a hardcoded list.
    return Promise.resolve([
        'default',
        'data_analyst',
        'devops',
        'docs',
        'finops',
        'knowledge_engineer',
        'meta_analyzer',
        'predictive_analyst',
        'reflector',
        'security_analyst',
    ]);
};

export const listWorkflows = async (): Promise<Workflow[]> => {
    const response = await fetch(`${API_BASE_URL}/user-workflows`, {
        headers: {
            'Content-Type': 'application/json',
        },
    });
    if (!response.ok) throw new Error("Failed to list workflows");
    return response.json();
};

export const getWorkflow = async (workflowId: string): Promise<Workflow> => {
    const response = await fetch(`${API_BASE_URL}/user-workflows/${workflowId}`, {
        headers: {
            'Content-Type': 'application/json',
        },
    });
    if (!response.ok) throw new Error("Workflow not found");
    return response.json();
};

export const runWorkflow = async (workflowId: string): Promise<void> => {
    const response = await fetch(`${API_BASE_URL}/user-workflows/${workflowId}/run`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to run workflow');
    }
}; 