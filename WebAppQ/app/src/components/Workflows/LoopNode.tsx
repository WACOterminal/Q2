// WebAppQ/app/src/components/Workflows/LoopNode.tsx
import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';

export const LoopNode: React.FC<NodeProps> = ({ data }) => {
  return (
    <div style={{
      padding: '10px 20px',
      border: '1px dashed #777',
      borderRadius: '5px',
      background: '#f8f8f8',
      width: 150,
      textAlign: 'center'
    }}>
      <Handle type="target" position={Position.Top} />
      <div>
        <strong>{data.label || 'Loop'}</strong>
      </div>
      <div style={{ fontSize: '12px', color: '#555', marginTop: '5px' }}>
        {data.condition || 'true'}
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        id="body"
        style={{ left: '50%', background: '#5bc0de' }}
      />
    </div>
  );
}; 