// WebAppQ/app/src/components/Workflows/WorkflowBuilder.tsx
import React, { useState, useCallback } from 'react';
import ReactFlow, {
  ReactFlowProvider,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  Connection,
  Edge,
  Node,
  isEdge,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { saveWorkflow } from '../../services/workflowBuilderAPI';
import { Workflow, WorkflowTask, ApprovalBlock, TaskBlock } from '../../services/types';
import { NodeDetailPanel } from './NodeDetailPanel';
import { ConditionalNode } from './ConditionalNode';
import { LoopNode } from './LoopNode';
import { ParallelNode } from './ParallelNode';
import './WorkflowBuilder.css';
import { Tooltip } from 'react-tooltip';
import 'react-tooltip/dist/react-tooltip.css';

const initialNodes: Node[] = [
  { id: '1', type: 'input', data: { label: 'Start' }, position: { x: 250, y: 5 } },
];

const nodeTypes = {
    conditional: ConditionalNode,
    loop: LoopNode,
    parallel: ParallelNode,
};

const Sidebar = () => {
    const onDragStart = (event: React.DragEvent, nodeType: string) => {
        event.dataTransfer.setData('application/reactflow', nodeType);
        event.dataTransfer.effectAllowed = 'move';
    };

    return (
        <aside style={{ borderRight: '1px solid #eee', padding: '15px', fontSize: '12px' }}>
            <div className="description">You can drag these nodes to the pane on the right.</div>
            <div 
                className="dndnode" 
                onDragStart={(event) => onDragStart(event, 'default')} 
                draggable
                data-tooltip-id="task-node-tooltip"
                data-tooltip-content="A standard task to be executed by an agent."
            >
                Task Node
            </div>
            <div 
                className="dndnode dndnode-output" 
                onDragStart={(event) => onDragStart(event, 'output')} 
                draggable
                data-tooltip-id="approval-node-tooltip"
                data-tooltip-content="Pauses the workflow and waits for a manual approval."
            >
                Approval Node
            </div>
            <div 
                className="dndnode dndnode-conditional" 
                onDragStart={(event) => onDragStart(event, 'conditional')} 
                draggable
                data-tooltip-id="conditional-node-tooltip"
                data-tooltip-content="Executes different branches based on a condition."
            >
                Conditional Node
            </div>
            <div 
                className="dndnode dndnode-loop" 
                onDragStart={(event) => onDragStart(event, 'loop')} 
                draggable
                data-tooltip-id="loop-node-tooltip"
                data-tooltip-content="Executes a set of tasks repeatedly."
            >
                Loop Node
            </div>
            <div 
                className="dndnode dndnode-parallel" 
                onDragStart={(event) => onDragStart(event, 'parallel')} 
                draggable
                data-tooltip-id="parallel-node-tooltip"
                data-tooltip-content="Executes multiple branches concurrently."
            >
                Parallel Node
            </div>
        </aside>
    );
};

let id = 2;
const getId = () => `${id++}`;

const transformFlowToWorkflow = (nodes: Node[], edges: Edge[]): Partial<Workflow> => {

    const buildTaskTree = (startNodeId: string): TaskBlock[] => {
        const tasks: TaskBlock[] = [];
        const visited = new Set<string>();

        const findNodeById = (id: string) => nodes.find(n => n.id === id);

        const traverse = (nodeId: string) => {
            if (visited.has(nodeId) || !nodeId) return;
            visited.add(nodeId);

            const node = findNodeById(nodeId);
            if (!node || node.type === 'input') return;

            const dependencies = edges.filter(edge => edge.target === node.id).map(edge => edge.source);
            let task: TaskBlock | null = null;

            if (node.type === 'output') { // Approval Node
                task = {
                    task_id: node.id,
                    type: 'approval',
                    message: node.data.message || 'Approval Required',
                    status: 'PENDING',
                    dependencies: dependencies,
                };
            } else if (node.type === 'conditional') {
                const trueEdge = edges.find(edge => edge.source === node.id && edge.sourceHandle === 'true');
                const falseEdge = edges.find(edge => edge.source === node.id && edge.sourceHandle === 'false');

                const trueTasks = trueEdge ? buildTaskTree(trueEdge.target) : [];
                const falseTasks = falseEdge ? buildTaskTree(falseEdge.target) : [];

                task = {
                    task_id: node.id,
                    type: 'conditional',
                    status: 'PENDING',
                    dependencies: dependencies,
                    branches: [
                        {
                            condition: node.data.condition || 'true == true',
                            tasks: trueTasks,
                        },
                        {
                            condition: 'default', // The 'else' case
                            tasks: falseTasks,
                        }
                    ]
                };
            } else if (node.type === 'loop') {
                const bodyEdge = edges.find(edge => edge.source === node.id && edge.sourceHandle === 'body');
                const bodyTasks = bodyEdge ? buildTaskTree(bodyEdge.target) : [];

                task = {
                    task_id: node.id,
                    type: 'loop',
                    status: 'PENDING',
                    dependencies: dependencies,
                    condition: node.data.condition || 'true',
                    tasks: bodyTasks,
                    max_iterations: 10, // Default max iterations
                };
            } else if (node.type === 'parallel') {
                const branchEdges = edges.filter(edge => edge.source === node.id);
                const branches = branchEdges.map(edge => buildTaskTree(edge.target));

                task = {
                    task_id: node.id,
                    type: 'parallel',
                    status: 'PENDING',
                    dependencies: dependencies,
                    branches: branches,
                };
            } else { // Default Task Node
                task = {
                    task_id: node.id,
                    type: 'task',
                    agent_personality: node.data.agent_personality || 'default',
                    prompt: node.data.prompt || '',
                    status: 'PENDING',
                    dependencies: dependencies,
                };
            }

            if(task) {
                tasks.push(task);
            }

            const outgoingEdges = edges.filter(edge => edge.source === node.id);
            outgoingEdges.forEach(edge => traverse(edge.target));
        };
        
        traverse(startNodeId);
        return tasks;
    };

    const startNode = nodes.find(node => node.type === 'input');
    const firstTaskEdge = edges.find(edge => edge.source === startNode?.id);
    const initialTasks = firstTaskEdge ? buildTaskTree(firstTaskEdge.target) : [];

    return {
        original_prompt: "User-created workflow",
        tasks: initialTasks,
        shared_context: {},
    };
};

export const WorkflowBuilder: React.FC = () => {
    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);
    const [selectedNode, setSelectedNode] = useState<Node | null>(null);

    const onConnect = useCallback(
        (params: Edge | Connection) => {
            let newEdge: Edge;
            if (isEdge(params)) {
                newEdge = params;
            } else {
                newEdge = {
                    id: `${params.source}-${params.target}`,
                    source: params.source!,
                    target: params.target!,
                };
            }
            
            const sourceNode = nodes.find(node => node.id === newEdge.source);
            if (sourceNode?.type === 'conditional') {
                newEdge.label = newEdge.sourceHandle === 'true' ? 'True' : 'False';
            }
            setEdges((eds) => addEdge(newEdge, eds));
        },
        [setEdges, nodes]
    );

    const onNodeDataChange = (nodeId: string, data: any) => {
        setNodes((nds) =>
            nds.map((node) => {
                if (node.id === nodeId) {
                    // it's important to create a new object here to trigger a re-render
                    node.data = { ...node.data, ...data };
                }
                return node;
            })
        );
    };

    const onDragOver = useCallback((event: React.DragEvent) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }, []);

    const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
        setSelectedNode(node);
    }, []);

    const onDrop = useCallback(
        (event: React.DragEvent) => {
            event.preventDefault();

            const type = event.dataTransfer.getData('application/reactflow');
            if (typeof type === 'undefined' || !type) {
                return;
            }

            const position = reactFlowInstance.screenToFlowPosition({
                x: event.clientX,
                y: event.clientY,
            });

            let newNode;

            if (type === 'conditional') {
                newNode = {
                    id: getId(),
                    type,
                    position,
                    data: { condition: 'true == true', label: 'If...' },
                };
            } else if (type === 'loop') {
                newNode = {
                    id: getId(),
                    type,
                    position,
                    data: { condition: 'true', label: 'Loop' },
                };
            } else if (type === 'parallel') {
                newNode = {
                    id: getId(),
                    type,
                    position,
                    data: { label: 'Parallel' },
                };
            } else {
                 newNode = {
                    id: getId(),
                    type,
                    position,
                    data: { label: `${type} node` },
                };
            }


            setNodes((nds) => nds.concat(newNode));
        },
        [reactFlowInstance, setNodes],
    );

    const onSave = async () => {
        const workflowPayload = transformFlowToWorkflow(nodes, edges);
        try {
            await saveWorkflow(workflowPayload);
            alert('Workflow saved successfully!');
        } catch (error) {
            console.error(error);
            alert('Failed to save workflow.');
        }
    };
    
    return (
        <div className="dndflow">
            <ReactFlowProvider>
                <div style={{ display: 'flex', flexDirection: 'row', height: '100%' }}>
                    <Sidebar />
                    <div className="reactflow-wrapper" style={{ flexGrow: 1, height: 'calc(100vh - 150px)' }}>
                        <ReactFlow
                            nodes={nodes}
                            edges={edges}
                            onNodesChange={onNodesChange}
                            onEdgesChange={onEdgesChange}
                            onConnect={onConnect}
                            onInit={setReactFlowInstance}
                            onDrop={onDrop}
                            onDragOver={onDragOver}
                            onNodeClick={onNodeClick}
                            nodeTypes={nodeTypes}
                            fitView
                        >
                            <Controls />
                            <Background />
                        </ReactFlow>
                    </div>
                    <NodeDetailPanel selectedNode={selectedNode} onNodeDataChange={onNodeDataChange} />
                </div>
                <div style={{ padding: '10px', borderTop: '1px solid #eee', textAlign: 'right' }}>
                    <button onClick={onSave} style={{ padding: '10px 20px', fontSize: '16px', background: '#4CAF50', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                        Save Workflow
                    </button>
                </div>
                <Tooltip id="task-node-tooltip" />
                <Tooltip id="approval-node-tooltip" />
                <Tooltip id="conditional-node-tooltip" />
                <Tooltip id="loop-node-tooltip" />
                <Tooltip id="parallel-node-tooltip" />
            </ReactFlowProvider>
        </div>
    );
}; 