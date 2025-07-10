// WebAppQ/app/src/components/Dashboard/ModelTestsWidget.tsx
import React, { useState, useEffect } from 'react';
import { Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, CircularProgress, Alert, Chip } from '@mui/material';
import { ModelTest, getModelTests } from '../../services/dashboardAPI';

export const ModelTestsWidget: React.FC = () => {
    const [tests, setTests] = useState<ModelTest[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchModelTests = async () => {
            try {
                setLoading(true);
                const testData = await getModelTests();
                setTests(testData);
                setError(null);
            } catch (err) {
                setError('Failed to fetch model test data.');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        fetchModelTests();
    }, []);

    const getStatusChipColor = (status: string) => {
        switch (status) {
            case 'PASS':
                return 'success';
            case 'FAIL':
                return 'error';
            case 'RUNNING':
                return 'info';
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
        <Paper sx={{ p: 2, height: '100%', minWidth: 600 }}>
            <Typography variant="h6" gutterBottom>Model Quality</Typography>
            <TableContainer>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell>Model Name</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell align="right">Accuracy</TableCell>
                            <TableCell>Last Run</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {tests.map((test) => (
                            <TableRow key={test.test_id}>
                                <TableCell component="th" scope="row">
                                    {test.model_name}
                                </TableCell>
                                <TableCell>
                                    <Chip label={test.status} color={getStatusChipColor(test.status)} size="small" />
                                </TableCell>
                                <TableCell align="right">{(test.accuracy * 100).toFixed(1)}%</TableCell>
                                <TableCell>{new Date(test.last_run).toLocaleString()}</TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Paper>
    );
}; 