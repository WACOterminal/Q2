import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, CircularProgress, Box, Button, Paper } from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { triggerWorkflow } from '../../services/managerAPI'; // Assuming this exists

// This would be a proper charting library in a real app
const MockChart = () => <Box height={150} bgcolor="#f0f0f0" display="flex" alignItems="center" justifyContent="center"><Typography variant="caption">Training Loss Chart Area</Typography></Box>;

export function QGANMonitorWidget() {
    const [isLoading, setIsLoading] = useState(false);
    const [status, setStatus] = useState('Idle');
    const [generatedData, setGeneratedData] = useState<any[] | null>(null);

    const handleRunExperiment = async () => {
        setIsLoading(true);
        setStatus('Triggering workflow...');
        try {
            const result = await triggerWorkflow('wf_qgan_data_generation');
            // In a real app, we would now poll the workflow instance for its status and final result.
            // For this simulation, we'll just move to a "complete" state after a delay.
            setStatus(`Training QGAN... (Task ID: ${result.task_id})`);
            setTimeout(() => {
                setStatus('Generation Complete');
                setGeneratedData([
                    { feature1: 0.12, feature2: 0.88 },
                    { feature1: 0.91, feature2: 0.05 },
                    { feature1: 0.45, feature2: 0.55 },
                ]);
                setIsLoading(false);
            }, 8000); // Simulate an 8-second training/generation time
        } catch (error) {
            setStatus(`Error: ${(error as Error).message}`);
            setIsLoading(false);
        }
    };

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="h6">Quantum Generative Adversarial Network</Typography>
                    <Button variant="contained" startIcon={<AutoAwesomeIcon />} onClick={handleRunExperiment} disabled={isLoading}>
                        Run Experiment
                    </Button>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Status: {status}
                </Typography>
                
                {isLoading && status.startsWith('Training') && <MockChart />}

                {generatedData && (
                    <Paper variant="outlined" sx={{ mt: 2, p: 2 }}>
                        <Typography variant="subtitle1">Generated Data Samples:</Typography>
                        <pre>{JSON.stringify(generatedData, null, 2)}</pre>
                    </Paper>
                )}
            </CardContent>
        </Card>
    );
} 