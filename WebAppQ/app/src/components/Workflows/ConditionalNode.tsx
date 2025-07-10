// WebAppQ/app/src/components/Workflows/ConditionalNode.tsx
import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

export const ConditionalNode: React.FC<NodeProps> = ({ data }) => {
  return (
    <div style={{
      padding: '10px 20px',
      border: '1px solid #777',
      borderRadius: '5px',
      background: '#f0f0f0',
      width: 150,
      textAlign: 'center'
    }}>
      <Handle type="target" position={Position.Top} />
      <div>
        <strong>{data.label}</strong>
      </div>
      <div style={{ fontSize: '12px', color: '#555', marginTop: '5px' }}>
        {data.condition}
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        id="true"
        style={{ left: '25%', background: '#5cb85c' }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="false"
        style={{ left: '75%', background: '#d9534f' }}
      />
    </div>
  );
}; 