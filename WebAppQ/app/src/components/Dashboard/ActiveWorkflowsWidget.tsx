// WebAppQ/app/src/components/Dashboard/ActiveWorkflowsWidget.tsx
import React, { useState, useEffect } from 'react';
import { Paper, Typography, List, ListItem, ListItemText, Chip, CircularProgress, Alert } from '@mui/material';
import { Workflow } from '../../services/types';
import { getActiveWorkflows } from '../../services/dashboardAPI';
import { Link } from 'react-router-dom';

export const ActiveWorkflowsWidget: React.FC = () => {
    const [workflows, setWorkflows] = useState<Workflow[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchWorkflows = async () => {
            try {
                setLoading(true);
                const activeWorkflows = await getActiveWorkflows();
                setWorkflows(activeWorkflows);
                setError(null);
            } catch (err) {
                setError('Failed to fetch active workflows.');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchWorkflows();
    }, []);

    const getStatusChipColor = (status: string) => {
        switch (status) {
            case 'RUNNING':
                return 'primary';
            case 'PENDING_APPROVAL':
                return 'warning';
            case 'FAILED':
                return 'error';
            default:
                return 'default';
        }
    };

    if (loading) {
        return <CircularProgress />;
    }

    if (error) {
        return <Alert severity="error">{error}</Alert>;
    }

    return (
        <Paper sx={{ p: 2, height: '100%', minWidth: 400 }}>
            <Typography variant="h6" gutterBottom>Active Workflows</Typography>
            <List>
                {workflows.length === 0 && <ListItem><ListItemText primary="No active workflows." /></ListItem>}
                {workflows.map(wf => (
                    <ListItem 
                        key={wf.workflow_id}
                        component={Link}
                        to={`/workflows/${wf.workflow_id}`}
                        sx={{ textDecoration: 'none', color: 'inherit' }}
                    >
                        <ListItemText 
                            primary={wf.original_prompt}
                            secondary={`ID: ${wf.workflow_id}`}
                        />
                        <Chip label={wf.status} color={getStatusChipColor(wf.status)} size="small" />
                    </ListItem>
                ))}
            </List>
        </Paper>
    );
}; 