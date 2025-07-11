import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, CircularProgress, List, ListItem, ListItemText, Chip, Box } from '@mui/material';
import { Warning, CheckCircle } from '@mui/icons-material';

// Placeholder data - in a real implementation, this would come from an API
const placeholderScanResult = {
    id: "scan-123",
    timestamp: new Date().toISOString(),
    status: "Completed",
    summary: "Scan completed on 8 services.",
    high_severity_findings: [
        {
            id: "finding-1",
            service: "UserProfileQ",
            file: "/app/UserProfileQ/app/core/cassandra_client.py",
            line: 45,
            description: "Use of `eval` is insecure.",
            ticket_id: "OP-1234"
        },
        {
            id: "finding-2",
            service: "H2M",
            file: "/app/H2M/app/services/feedback_processor.py",
            line: 88,
            description: "Hardcoded password found.",
            ticket_id: "OP-1235"
        }
    ]
};

const noFindingsResult = {
    id: "scan-124",
    timestamp: new Date().toISOString(),
    status: "Completed",
    summary: "Scan completed on 8 services.",
    high_severity_findings: []
};


export function SecurityScanWidget() {
    const [scan, setScan] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Simulate fetching data
        setTimeout(() => {
            // Randomly choose between a result with findings and one without
            setScan(Math.random() > 0.5 ? placeholderScanResult : noFindingsResult);
            setLoading(false);
        }, 2000);
    }, []);

    const hasFindings = scan?.high_severity_findings?.length > 0;

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="h6" gutterBottom>
                        Latest Security Scan
                    </Typography>
                    {scan && (
                         <Chip
                            icon={hasFindings ? <Warning /> : <CheckCircle />}
                            label={hasFindings ? `${scan.high_severity_findings.length} High-Severity Findings` : "No Critical Issues"}
                            color={hasFindings ? "error" : "success"}
                        />
                    )}
                </Box>
                {loading ? (
                    <CircularProgress />
                ) : (
                    <>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                           Last scan: {new Date(scan.timestamp).toLocaleString()}
                        </Typography>
                        <List dense>
                            {hasFindings ? (
                                scan.high_severity_findings.map((finding: any) => (
                                    <ListItem key={finding.id}>
                                        <ListItemText
                                            primary={`${finding.description} in ${finding.service}`}
                                            secondary={`File: ${finding.file}:${finding.line} | Ticket: ${finding.ticket_id}`}
                                        />
                                    </ListItem>
                                ))
                            ) : (
                                <ListItem>
                                    <ListItemText primary="No new high-severity vulnerabilities were detected in the last scan." />
                                </ListItem>
                            )}
                        </List>
                    </>
                )}
            </CardContent>
        </Card>
    );
} 