import React, { useState, useEffect } from 'react';
import { useParams, Link as RouterLink } from 'react-router-dom';
import { Container, Typography, Paper, Box, CircularProgress, Alert, Button, Divider, List, ListItem, ListItemText } from '@mui/material';
import { getVetoedWorkflowDetails } from '../../services/managerAPI'; // To be created
import { Workflow } from '../../services/types'; // Assuming this type exists

export function VetoReviewPage() {
    const { workflowId } = useParams<{ workflowId: string }>();
    const [workflow, setWorkflow] = useState<Workflow | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!workflowId) return;
        const fetchDetails = async () => {
            try {
                setLoading(true);
                const data = await getVetoedWorkflowDetails(workflowId);
                setWorkflow(data);
            } catch (err) {
                setError((err as Error).message);
            } finally {
                setLoading(false);
            }
        };
        fetchDetails();
    }, [workflowId]);

    if (loading) return <CircularProgress />;
    if (error) return <Alert severity="error">{error}</Alert>;
    if (!workflow) return <Typography>Vetoed workflow not found.</Typography>;
    
    // The veto reason is stored in the final_result of the workflow
    const vetoReason = workflow.final_result || "No reason provided.";

    return (
        <Container maxWidth="lg" sx={{ mt: 4 }}>
            <Paper sx={{ p: 3 }}>
                <Typography variant="h4" gutterBottom>
                    Ethical Veto Review
                </Typography>
                <Typography variant="subtitle1" gutterBottom>
                    Workflow ID: {workflow.workflow_id}
                </Typography>
                <Divider sx={{ my: 2 }} />

                <Alert severity="warning" variant="filled">
                    <Typography><strong>Reason for Veto:</strong> {vetoReason}</Typography>
                </Alert>

                <Box mt={3}>
                    <Typography variant="h6">Original Goal</Typography>
                    <Paper variant="outlined" sx={{ p: 2, mt: 1, bgcolor: 'grey.100' }}>
                        <Typography fontFamily="monospace">{workflow.original_prompt}</Typography>
                    </Paper>
                </Box>
                
                <Box mt={3}>
                    <Button
                        component={RouterLink}
                        to={`/workflows/${workflow.workflow_id}`}
                        variant="contained"
                    >
                        View Full Workflow Details
                    </Button>
                </Box>
            </Paper>
        </Container>
    );
} 