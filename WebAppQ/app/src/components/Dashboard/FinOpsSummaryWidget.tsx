import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, CircularProgress, Box, List, ListItem, ListItemText, Divider } from '@mui/material';
import MonetizationOnIcon from '@mui/icons-material/MonetizationOn';
import { getFinOpsReport } from '../../services/managerAPI'; // Assuming this function exists

export function FinOpsSummaryWidget() {
    const [report, setReport] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchReport = async () => {
            try {
                setLoading(true);
                const data = await getFinOpsReport();
                setReport(data);
            } catch (err) {
                setError((err as Error).message);
            } finally {
                setLoading(false);
            }
        };
        fetchReport();
    }, []);

    const hasIssues = report?.potential_issues?.length > 0;

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                 <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <MonetizationOnIcon sx={{ mr: 1, color: 'success.main' }} />
                    <Typography variant="h6">
                        Daily FinOps Summary
                    </Typography>
                </Box>
                {loading ? (
                    <CircularProgress />
                ) : error ? (
                    <Typography color="error">Failed to load report: {error}</Typography>
                ) : report ? (
                    <Box>
                        <Typography variant="subtitle1" gutterBottom>
                            {report.overall_summary}
                        </Typography>
                        <Divider sx={{ my: 1 }} />
                        <Typography variant="h6" sx={{ mt: 2, fontSize: '1rem', color: hasIssues ? 'error.main' : 'text.primary' }}>
                            {hasIssues ? "Potential Issues Identified" : "No Cost Anomalies Detected"}
                        </Typography>
                        {hasIssues && (
                            <List dense>
                                {report.potential_issues.map((issue: any, index: number) => (
                                    <ListItem key={index}>
                                        <ListItemText primary={`${issue.type}: ${issue.service}`} secondary={issue.details} />
                                    </ListItem>
                                ))}
                            </List>
                        )}
                        
                        {report.venture_pnl && (
                            <>
                                <Divider sx={{ my: 2 }} />
                                <Typography variant="h6" sx={{ mt: 2, fontSize: '1rem' }}>
                                    Autonomous Ventures P&L
                                </Typography>
                                <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                                    Net Profit: ${report.venture_pnl.net_profit_usd.toFixed(2)}
                                </Typography>
                            </>
                        )}
                    </Box>
                ) : (
                     <Typography variant="body1" color="text.secondary">
                        No FinOps report available.
                    </Typography>
                )}
            </CardContent>
        </Card>
    );
} 