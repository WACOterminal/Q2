import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, CircularProgress, List, ListItem, ListItemText, Divider, Paper } from '@mui/material';

// Placeholder data - in a real implementation, this would come from an API
const placeholderReports = [
    {
        id: "rca-1",
        service: "userprofile-q",
        summary: "Service experienced cascading failures due to database connection exhaustion.",
        timestamp: new Date().toISOString(),
        evidence: [
            "Metrics: P99 latency spiked to 5000ms.",
            "Logs: Repeated 'FATAL: sorry, too many clients already' errors.",
            "K8s Events: Pods began restarting with 'CrashLoopBackOff' status."
        ],
        recommendation: "Increase the max connections limit on the PostgreSQL database and implement connection pooling in the service."
    },
    {
        id: "rca-2",
        service: "quantumpulse-api",
        summary: "Memory leak in the inference cache led to OOMKilled events.",
        timestamp: new Date(Date.now() - 3600 * 1000).toISOString(),
        evidence: [
            "Metrics: Memory usage grew linearly over 1 hour before dropping sharply.",
            "Logs: No significant errors, but request volume was high.",
            "K8s Events: Multiple pods were terminated with reason 'OOMKilled'."
        ],
        recommendation: "Analyze the inference caching mechanism for memory leaks and consider adding a cache eviction policy."
    }
];

export function RCAReportsWidget() {
    const [reports, setReports] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Simulate fetching data
        setTimeout(() => {
            setReports(placeholderReports);
            setLoading(false);
        }, 1500);
    }, []);

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Typography variant="h6" gutterBottom>
                    Latest Root Cause Analysis Reports
                </Typography>
                {loading ? (
                    <CircularProgress />
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