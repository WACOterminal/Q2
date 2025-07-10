// WebAppQ/app/src/services/elasticsearch.ts
import { Client } from '@elastic/elasticsearch';

const ELASTICSEARCH_URL = process.env.REACT_APP_ELASTICSEARCH_URL || 'http://localhost:9200';

const client = new Client({ node: ELASTICSEARCH_URL });

export interface WorkflowAnalyticData {
    agent_id: string;
    success_count: number;
    failure_count: number;
    timestamp: number;
}

export const fetchWorkflowAnalytics = async (): Promise<WorkflowAnalyticData[]> => {
    try {
        const response = await client.search({
            index: 'workflow_analytics',
            body: {
                size: 1000, // Get the last 1000 data points
                sort: [
                    { "timestamp": { "order": "desc" } }
                ],
                query: {
                    match_all: {}
                }
            }
        });

        return response.hits.hits.map((hit: any) => hit._source as WorkflowAnalyticData);
    } catch (error) {
        console.error('Error fetching workflow analytics:', error);
        return [];
    }
}; 