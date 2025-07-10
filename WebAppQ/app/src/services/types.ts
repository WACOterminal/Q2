// WebAppQ/app/src/services/types.ts

export interface SearchQuery {
    query: string;
    session_id?: string;
}

export interface VectorStoreResult {
    source: string;
    content: string;
    score: number;
    metadata: Record<string, any>;
}

export interface KGNode {
    id: string;
    label: string;
    properties: Record<string, any>;
}

export interface KGEdge {
    source: string;
    target: string;
    label: string;
}

export interface KnowledgeGraphResult {
    nodes: KGNode[];
    edges: KGEdge[];
}

export interface SearchResponse {
    ai_summary: string;
    vector_results: VectorStoreResult[];
    knowledge_graph_result: KnowledgeGraphResult | null;
    model_version?: string;
}

export interface FeedbackEvent {
    reference_id: string;
    context: string;
    score: number;
    prompt?: string;
    feedback_text?: string;
    model_version?: string;
}

export interface UserCreate {
    username: string;
    email: string;
    password: string;
    first_name?: string;
    last_name?: string;
}

// --- Workflow & Task Models ---

export type TaskStatus = "PENDING" | "DISPATCHED" | "RUNNING" | "COMPLETED" | "FAILED" | "PENDING_APPROVAL" | "CANCELLED";
export type WorkflowStatus = "PENDING" | "RUNNING" | "PAUSED" | "COMPLETED" | "FAILED" | "PENDING_CLARIFICATION" | "CANCELLED";

export interface WorkflowTask {
    task_id: string;
    type: 'task';
    agent_personality: string;
    prompt: string;
    status: TaskStatus;
    dependencies: string[];
    result?: string;
}

export interface ConditionalBlock {
    task_id: string;
    type: 'conditional';
    status: TaskStatus;
    dependencies: string[];
    branches: {
        condition: string;
        tasks: TaskBlock[];
    }[];
}

export interface LoopBlock {
    task_id: string;
    type: 'loop';
    status: TaskStatus;
    dependencies: string[];
    condition: string;
    tasks: TaskBlock[];
    max_iterations: number;
}

export interface ParallelBlock {
    task_id: string;
    type: 'parallel';
    status: TaskStatus;
    dependencies: string[];
    branches: TaskBlock[][];
}

export interface ApprovalBlock {
    task_id: string;
    type: 'approval';
    status: TaskStatus;
    dependencies: string[];
    message: string;
}

export type TaskBlock = WorkflowTask | ConditionalBlock | LoopBlock | ApprovalBlock | ParallelBlock;

export interface Workflow {
    workflow_id: string;
    original_prompt: string;
    status: WorkflowStatus;
    tasks: TaskBlock[];
    shared_context: Record<string, any>;
    final_result?: string;
} 