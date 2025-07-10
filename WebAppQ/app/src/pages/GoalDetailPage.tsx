
import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Box, Typography, CircularProgress, Alert, Paper, Button, Stack } from '@mui/material';
import { getWorkflow, getGoal, controlWorkflow } from '../../services/managerAPI';
import { WorkflowVisualizer } from '../../components/WorkflowVisualizer/WorkflowVisualizer';
import { Workflow } from '../../services/types';

// Define a type for the Goal object we expect from the API
interface Goal {
    id: number;
    subject: string;
    description: { raw: string }; 
    status: string;
    _links: {
        customField1?: { href: string, title: string }; 
    }
}

const GoalDetailPage: React.FC = () => {
    const { goalId } = useParams<{ goalId: string }>();
    const [goal, setGoal] = useState<Goal | null>(null);
    const [workflow, setWorkflow] = useState<Workflow | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    const fetchGoalDetails = useCallback(async () => {
        if (!goalId) return;
        try {
            setLoading(true);
            const goalData = await getGoal(goalId);
            setGoal(goalData);

            const workflowId = goalData?._links?.customField1?.title;
            if (workflowId) {
                const workflowData = await getWorkflow(workflowId);
                setWorkflow(workflowData);
            }
            setError(null);
        } catch (err: any) {
            setError(err.message || 'An unexpected error occurred.');
            console.error("Failed to fetch goal details:", err);
        } finally {
            setLoading(false);
        }
    }, [goalId]);

    useEffect(() => {
        fetchGoalDetails();
    }, [fetchGoalDetails]);

    const handleWorkflowControl = async (action: 'pause' | 'resume' | 'cancel') => {
        if (!workflow) {
            alert("No workflow associated with this goal to control.");
            return;
        }
        try {
            await controlWorkflow(workflow.workflow_id, action);
            alert(`Workflow action '${action}' successful.`);
            // Refresh data after action to show new status
            await fetchGoalDetails(); 
        } catch (err: any) {
            alert(`Failed to perform action '${action}': ${err.message}`);
        }
    };

    if (loading) {
        return <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}><CircularProgress /></Box>;
    }

    if (error) {
        return <Alert severity="error">{error}</Alert>;
    }

    if (!goal) {
        return <Alert severity="info">Goal not found.</Alert>;
    }

    return (
        <Box sx={{ p: 3 }}>
            <Paper sx={{ p: 3, mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box>
                        <Typography variant="h4" gutterBottom>{goal.subject}</Typography>
                        <Typography variant="h6" color="text.secondary">
                            Status: {goal.status} {workflow && `| Workflow: ${workflow.status}`}
                        </Typography>
                    </Box>
                    {workflow && (
                        <Stack direction="row" spacing={1}>
                            <Button 
                                variant="outlined" 
                                onClick={() => handleWorkflowControl('pause')}
                                disabled={workflow.status !== 'RUNNING'}
                            >
                                Pause
                            </Button>
                            <Button 
                                variant="outlined" 
                                onClick={() => handleWorkflowControl('resume')}
                                disabled={workflow.status !== 'PAUSED'}
                            >
                                Resume
                            </Button>
                            <Button 
                                variant="contained" 
                                color="error" 
                                onClick={() => handleWorkflowControl('cancel')}
                                disabled={workflow.status === 'COMPLETED' || workflow.status === 'FAILED' || workflow.status === 'CANCELLED'}
                            >
                                Cancel
                            </Button>
                        </Stack>
                    )}
                </Box>
                <Typography variant="body1" sx={{ mt: 2 }} dangerouslySetInnerHTML={{ __html: goal.description.raw }} />
            </Paper>
            
            <Typography variant="h5" gutterBottom>Associated Workflow</Typography>
            
            {workflow ? (
                <WorkflowVisualizer workflow={workflow} />
            ) : (
                <Alert severity="info">No associated workflow found for this goal.</Alert>
            )}
        </Box>
    );
};

export default GoalDetailPage; 