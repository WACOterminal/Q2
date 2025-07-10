// WebAppQ/app/src/components/Workflows/ParallelNode.tsx
import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

export const ParallelNode: React.FC<NodeProps> = ({ data }) => {
  return (
    <div style={{
      padding: '10px 20px',
      border: '2px solid #999',
      borderRadius: '5px',
      background: '#fafafa',
      width: 150,
      textAlign: 'center'
    }}>
      <Handle type="target" position={Position.Top} />
      <div>
        <strong>{data.label || 'Parallel'}</strong>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        id="branch1"
        style={{ left: '25%', background: '#f0ad4e' }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="branch2"
        style={{ left: '75%', background: '#f0ad4e' }}
      />
    </div>
  );
}; 