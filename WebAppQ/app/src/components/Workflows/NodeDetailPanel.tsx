// WebAppQ/app/src/components/Workflows/NodeDetailPanel.tsx
import React, { useState, useEffect } from 'react';
import { Node } from 'reactflow';
import { getAgentPersonalities } from '../../services/workflowBuilderAPI';

interface NodeDetailPanelProps {
  selectedNode: Node | null;
  onNodeDataChange: (nodeId: string, data: any) => void;
}

export const NodeDetailPanel: React.FC<NodeDetailPanelProps> = ({ selectedNode, onNodeDataChange }) => {
  const [availableAgents, setAvailableAgents] = useState<string[]>([]);

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        const agents = await getAgentPersonalities();
        setAvailableAgents(agents);
      } catch (error) {
        console.error("Failed to fetch agent personalities:", error);
        // Fallback to a default list in case of an error
        setAvailableAgents(['default', 'data_analyst', 'devops']);
      }
    };
    fetchAgents();
  }, []);

  if (!selectedNode) {
    return (
      <aside style={{ width: '250px', borderLeft: '1px solid #eee', padding: '15px' }}>
        <div className="description">Select a node to see its details.</div>
      </aside>
    );
  }

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = event.target;
    const newData = { ...selectedNode.data, [name]: value };
    onNodeDataChange(selectedNode.id, newData);
  };

  return (
    <aside style={{ width: '300px', borderLeft: '1px solid #eee', padding: '15px', fontSize: '12px' }}>
        <h3>Node Details</h3>
        <div><strong>ID:</strong> {selectedNode.id}</div>
        <div><strong>Type:</strong> {selectedNode.type}</div>
        <hr style={{margin: '15px 0'}}/>

        {selectedNode.type === 'conditional' && (
            <div style={{ marginTop: '10px' }}>
                <label>Condition:</label>
                <input
                    type="text"
                    name="condition"
                    value={selectedNode.data.condition || ''}
                    onChange={handleInputChange}
                    style={{ width: '100%', marginTop: '5px' }}
                />
            </div>
        )}

        {selectedNode.type === 'loop' && (
            <div style={{ marginTop: '10px' }}>
                <label>Loop Condition:</label>
                <input
                    type="text"
                    name="condition"
                    value={selectedNode.data.condition || ''}
                    onChange={handleInputChange}
                    style={{ width: '100%', marginTop: '5px' }}
                />
            </div>
        )}

        {selectedNode.type === 'default' && (
            <>
                <div style={{ marginTop: '10px' }}>
                    <label>Agent Personality:</label>
                    <select 
                        name="agent_personality"
                        value={selectedNode.data.agent_personality || 'default'} 
                        onChange={handleInputChange}
                        style={{ width: '100%', marginTop: '5px' }}
                    >
                        {availableAgents.map(agent => (
                            <option key={agent} value={agent}>{agent}</option>
                        ))}
                    </select>
                </div>
                <div style={{ marginTop: '10px' }}>
                    <label>Prompt:</label>
                    <textarea 
                        name="prompt"
                        value={selectedNode.data.prompt || ''} 
                        onChange={handleInputChange} 
                        rows={4} 
                        style={{ width: '100%', marginTop: '5px' }}
                    />
                </div>
            </>
        )}

        {selectedNode.type === 'output' && (
            <div style={{ marginTop: '10px' }}>
                <label>Approval Message:</label>
                <textarea 
                    name="message"
                    value={selectedNode.data.message || ''} 
                    onChange={handleInputChange} 
                    rows={4} 
                    style={{ width: '100%', marginTop: '5px' }}
                />
            </div>
        )}
    </aside>
  );
}; 