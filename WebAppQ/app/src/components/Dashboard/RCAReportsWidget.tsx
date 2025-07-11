import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, CircularProgress, List, ListItem, ListItemText, Divider, Paper } from '@mui/material';
import { getRCAReports } from '../../services/managerAPI'; // Assuming this exists

export function RCAReportsWidget() {
    const [reports, setReports] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchReports = async () => {
            try {
                setLoading(true);
                const data = await getRCAReports();
                setReports(data);
            } catch (err) {
                setError((err as Error).message);
            } finally {
                setLoading(false);
            }
        };
        fetchReports();
    }, []);

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Typography variant="h6" gutterBottom>
                    Latest Root Cause Analysis Reports
                </Typography>
                {loading ? (
                    <CircularProgress />
                ) : error ? (
                    <Typography color="error">Failed to load reports: {error}</Typography>
                ) : (
                    <List dense>
                        {reports.map((report, index) => (
                            <Paper key={report.id} elevation={2} sx={{ mb: 2, p: 2 }}>
                                <Typography variant="subtitle1" component="div">
                                    <strong>{report.service}</strong> - {new Date(report.timestamp).toLocaleString()}
                                </Typography>
                                <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic', mb: 1 }}>
                                    {report.summary}
                                </Typography>
                                <Typography variant="body2"><strong>Recommendation:</strong> {report.recommendation}</Typography>
                            </Paper>
                        ))}
                    </List>
                )}
            </CardContent>
        </Card>
    );
} 