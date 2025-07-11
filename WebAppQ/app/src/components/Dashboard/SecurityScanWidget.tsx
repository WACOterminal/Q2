import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, CircularProgress, List, ListItem, ListItemText, Chip, Box } from '@mui/material';
import { Warning, CheckCircle } from '@mui/icons-material';
import { getSecurityReport } from '../../services/managerAPI'; // Assuming this exists

export function SecurityScanWidget() {
    const [report, setReport] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchReport = async () => {
            try {
                setLoading(true);
                const data = await getSecurityReport();
                setReport(data);
            } catch (err) {
                setError((err as Error).message);
            } finally {
                setLoading(false);
            }
        };
        fetchReport();
    }, []);

    const hasFindings = report?.high_severity_findings?.length > 0;

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="h6" gutterBottom>
                        Latest Security Scan
                    </Typography>
                    {report && (
                         <Chip
                            icon={hasFindings ? <Warning /> : <CheckCircle />}
                            label={hasFindings ? `${report.high_severity_findings.length} High-Severity Findings` : "No Critical Issues"}
                            color={hasFindings ? "error" : "success"}
                        />
                    )}
                </Box>
                {loading ? (
                    <CircularProgress />
                ) : error ? (
                    <Typography color="error">Failed to load report: {error}</Typography>
                ) : report ? (
                    <>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                           Status: {report.status}
                        </Typography>
                        <List dense>
                            {hasFindings ? (
                                report.high_severity_findings.map((finding: any) => (
                                    <ListItem key={finding.id}>
                                        <ListItemText
                                            primary={finding.description}
                                            secondary={`Ticket: ${finding.ticket_id}`}
                                        />
                                    </ListItem>
                                ))
                            ) : (
                                <ListItem>
                                    <ListItemText primary="No new high-severity vulnerabilities were detected." />
                                </ListItem>
                            )}
                        </List>
                    </>
                ) : (
                    <Typography>No security scan report available.</Typography>
                )}
            </CardContent>
        </Card>
    );
} 