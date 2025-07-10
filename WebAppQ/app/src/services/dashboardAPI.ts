// WebAppQ/app/src/services/dashboardAPI.ts
import { Workflow, WorkflowStatus, TaskStatus } from './types';

export interface AgentPerformance {
    agent_id: string;
    tasks_completed: number;
    tasks_failed: number;
    average_execution_time: number; // in seconds
}

export interface ModelTest {
    test_id: string;
    model_name: string;
    status: 'PASS' | 'FAIL' | 'RUNNING';
    accuracy: number;
    last_run: string;
}

export const getActiveWorkflows = async (): Promise<Workflow[]> => {
    return Promise.resolve([
        {
            workflow_id: 'wf-123-abc',
            original_prompt: 'Summarize the latest user feedback from the last 24 hours.',
            status: 'RUNNING' as WorkflowStatus,
            tasks: [
                { task_id: 'task-1', type: 'task', agent_personality: 'data_analyst', prompt: 'Fetch feedback', status: 'COMPLETED' as TaskStatus, dependencies: [] },
                { task_id: 'task-2', type: 'task', agent_personality: 'data_analyst', prompt: 'Summarize feedback', status: 'RUNNING' as TaskStatus, dependencies: ['task-1'] }
            ],
            shared_context: {},
        },
        {
            workflow_id: 'wf-456-def',
            original_prompt: 'Deploy the new frontend to staging.',
            status: 'PENDING_APPROVAL' as WorkflowStatus,
            tasks: [
                { task_id: 'task-3', type: 'task', agent_personality: 'devops', prompt: 'Run build script', status: 'COMPLETED' as TaskStatus, dependencies: [] },
                { task_id: 'task-4', type: 'approval', message: 'Ready to deploy to staging?', status: 'PENDING_APPROVAL' as TaskStatus, dependencies: ['task-3'] }
            ],
            shared_context: {},
        }
    ]);
};

export const getAgentPerformance = async (): Promise<AgentPerformance[]> => {
    return Promise.resolve([
        { agent_id: 'data_analyst', tasks_completed: 152, tasks_failed: 5, average_execution_time: 45 },
        { agent_id: 'devops', tasks_completed: 89, tasks_failed: 2, average_execution_time: 120 },
        { agent_id: 'knowledge_engineer', tasks_completed: 210, tasks_failed: 1, average_execution_time: 25 },
    ]);
};

export const getModelTests = async (): Promise<ModelTest[]> => {
    return Promise.resolve([
        { test_id: 'test-abc', model_name: 'gpt-4-turbo', status: 'PASS', accuracy: 0.92, last_run: '2024-07-29T10:00:00Z' },
        { test_id: 'test-def', model_name: 'claude-3-opus', status: 'PASS', accuracy: 0.95, last_run: '2024-07-29T10:05:00Z' },
        { test_id: 'test-ghi', model_name: 'llama-3-70b', status: 'FAIL', accuracy: 0.88, last_run: '2024-07-29T10:10:00Z' },
        { test_id: 'test-jkl', model_name: 'internal-finetune-v2', status: 'RUNNING', accuracy: 0.0, last_run: '2024-07-29T11:00:00Z' },
    ]);
}; 