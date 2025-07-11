import React, { useState } from 'react';
import {
    Dialog, DialogTitle, DialogContent, DialogActions, Button, TextField,
    Box, CircularProgress, Typography, Alert
} from '@mui/material';
import { generateWorkflowFromPrompt } from '../../services/managerAPI';
import { getTaskResult } from '../../services/managerAPI'; // Assuming this function exists to poll for task results

interface GenerateWorkflowModalProps {
    open: boolean;
    onClose: () => void;
    onWorkflowGenerated: (yaml: string) => void;
}

export function GenerateWorkflowModal({ open, onClose, onWorkflowGenerated }: GenerateWorkflowModalProps) {
    const [prompt, setPrompt] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleGenerate = async () => {
        if (!prompt) return;
        setIsLoading(true);
        setError(null);

        try {
            const { task_id } = await generateWorkflowFromPrompt(prompt);
            
            // Poll for the result
            const pollResult = async () => {
                try {
                    const result = await getTaskResult(task_id);
                    if (result.status === 'COMPLETED') {
                        // The final result from the agent is a JSON string with thought and result
                        const agentResult = JSON.parse(result.result);
                        onWorkflowGenerated(agentResult.result); // The YAML is in the 'result' field
                        setIsLoading(false);
                        onClose();
                    } else if (result.status === 'FAILED') {
                        throw new Error(result.result || 'Workflow generation failed.');
                    } else {
                        setTimeout(pollResult, 2000); // Poll every 2 seconds
                    }
                } catch (pollError) {
                    setError((pollError as Error).message);
                    setIsLoading(false);
                }
            };
            setTimeout(pollResult, 2000);

        } catch (initialError) {
            setError((initialError as Error).message);
            setIsLoading(false);
        }
    };

    return (
        <Dialog open={open} onClose={onClose} fullWidth maxWidth="md">
            <DialogTitle>Generate Workflow with AI</DialogTitle>
            <DialogContent>
                <Typography variant="body1" sx={{ mb: 2 }}>
                    Describe the workflow you want to create in plain English. The AI will generate the YAML for you.
                </Typography>
                <TextField
                    autoFocus
                    margin="dense"
                    id="name"
                    label="Workflow Description"
                    type="text"
                    fullWidth
                    variant="outlined"
                    multiline
                    rows={4}
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="e.g., 'Every hour, check the status of the webapp-q service and send a summary to the #devops channel in Zulip.'"
                />
                {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Cancel</Button>
                <Box sx={{ m: 1, position: 'relative' }}>
                    <Button
                        variant="contained"
                        onClick={handleGenerate}
                        disabled={isLoading}
                    >
                        Generate
                    </Button>
                    {isLoading && (
                        <CircularProgress
                            size={24}
                            sx={{
                                position: 'absolute',
                                top: '50%',
                                left: '50%',
                                marginTop: '-12px',
                                marginLeft: '-12px',
                            }}
                        />
                    )}
                </Box>
            </DialogActions>
        </Dialog>
    );
} 