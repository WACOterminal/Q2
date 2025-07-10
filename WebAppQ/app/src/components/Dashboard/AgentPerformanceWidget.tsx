// WebAppQ/app/src/components/Dashboard/AgentPerformanceWidget.tsx
import React, { useState, useEffect } from 'react';
import { Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, CircularProgress, Alert } from '@mui/material';
import { AgentPerformance, getAgentPerformance } from '../../services/dashboardAPI';

export const AgentPerformanceWidget: React.FC = () => {
    const [agents, setAgents] = useState<AgentPerformance[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchAgentPerformance = async () => {
            try {
                setLoading(true);
                const performanceData = await getAgentPerformance();
                setAgents(performanceData);
                setError(null);
            } catch (err) {
                setError('Failed to fetch agent performance data.');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchAgentPerformance();
    }, []);

    if (loading) {
        return <CircularProgress />;
    }

    if (error) {
        return <Alert severity="error">{error}</Alert>;
    }

    return (
        <Paper sx={{ p: 2, height: '100%', minWidth: 500 }}>
            <Typography variant="h6" gutterBottom>Agent Performance</Typography>
            <TableContainer>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell>Agent ID</TableCell>
                            <TableCell align="right">Completed</TableCell>
                            <TableCell align="right">Failed</TableCell>
                            <TableCell align="right">Avg. Time (s)</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {agents.map((agent) => (
                            <TableRow key={agent.agent_id}>
                                <TableCell component="th" scope="row">
                                    {agent.agent_id}
                                </TableCell>
                                <TableCell align="right">{agent.tasks_completed}</TableCell>
                                <TableCell align="right">{agent.tasks_failed}</TableCell>
                                <TableCell align="right">{agent.average_execution_time}</TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Paper>
    );
}; 