// WebAppQ/app/src/components/Workflows/ApprovalCard.tsx
import React from 'react';
import { Card, CardContent, Typography, Button, Box, Alert, AlertTitle } from '@mui/material';
import { DiffViewer } from './DiffViewer'; // A new component to create for showing diffs

interface ApprovalCardProps {
    workflowId: string;
    taskId: string;
    message: string;
    onApprove: (workflowId: string, taskId: string) => void;
    onReject: (workflowId: string, taskId: string) => void;
    isPromptModification?: boolean;
    originalPrompt?: string;
    suggestedPrompt?: string;
}

export const ApprovalCard: React.FC<ApprovalCardProps> = ({
    workflowId,
    taskId,
    message,
    onApprove,
    onReject,
    isPromptModification = false,
    originalPrompt = '',
    suggestedPrompt = '',
}) => {
    return (
        <Card sx={{ mb: 2 }}>
            <CardContent>
                {isPromptModification && (
                    <Alert severity="warning" sx={{ mb: 2 }}>
                        <AlertTitle>High-Privilege Action Required</AlertTitle>
                        This approval will modify an agent's core system prompt. Review carefully.
                    </Alert>
                )}

                <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                    {message}
                </Typography>

                {isPromptModification && (
                    <Box mt={2}>
                        <Typography variant="h6">Proposed Prompt Changes:</Typography>
                        <DiffViewer oldText={originalPrompt} newText={suggestedPrompt} />
                    </Box>
                )}

                <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                    <Button variant="outlined" color="error" onClick={() => onReject(workflowId, taskId)} sx={{ mr: 1 }}>
                        Reject
                    </Button>
                    <Button variant="contained" color="success" onClick={() => onApprove(workflowId, taskId)}>
                        Approve
                    </Button>
                </Box>
            </CardContent>
        </Card>
    );
}; 