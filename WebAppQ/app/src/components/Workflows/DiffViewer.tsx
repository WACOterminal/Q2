import React from 'react';
import { diffWordsWithSpace } from 'diff';
import { Box, Paper, Typography } from '@mui/material';

interface DiffViewerProps {
    oldText: string;
    newText: string;
}

export const DiffViewer: React.FC<DiffViewerProps> = ({ oldText, newText }) => {
    const diff = diffWordsWithSpace(oldText, newText);

    return (
        <Paper variant="outlined" sx={{ p: 2, mt: 1, whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
            {diff.map((part, index) => {
                const color = part.added ? 'success.main' :
                              part.removed ? 'error.main' : 'text.primary';
                const textDecoration = part.removed ? 'line-through' : 'none';
                
                return (
                    <Typography 
                        key={index} 
                        component="span" 
                        sx={{ color, textDecoration, bgcolor: part.added ? '#e6ffed' : part.removed ? '#ffeef0' : 'inherit' }}
                    >
                        {part.value}
                    </Typography>
                );
            })}
        </Paper>
    );
}; 