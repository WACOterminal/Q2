import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, CircularProgress, Box, List, ListItem, ListItemText, Divider } from '@mui/material';
import MonetizationOnIcon from '@mui/icons-material/MonetizationOn';

// Placeholder data - in a real implementation, this would come from an API
const placeholderReport = {
    report_id: "finops-2024-08-01",
    timestamp: new Date().toISOString(),
    overall_summary: "Total combined spend is $9850.55. One cost anomaly detected.",
    potential_issues: [
        {
            type: "Cost Spike",
            service: "QuantumPulse",
            details: "Service cost of $3201.10 is more than 2x the average service cost of $1329.89."
        }
    ],
    recommendation: "Manual review of flagged service costs is recommended."
};

const noIssuesReport = {
    report_id: "finops-2024-08-02",
    timestamp: new Date().toISOString(),
    overall_summary: "Total combined spend is $8123.45. No cost anomalies detected.",
    potential_issues: [],
    recommendation: "Costs are within expected parameters."
};

const placeholderVentureReport = {
    total_ventures: 12,
    total_revenue_usd: 1850.75,
    total_cost_usd: 740.30,
    net_profit_usd: 1110.45
};

export function FinOpsSummaryWidget() {
    const [report, setReport] = useState<any>(null);
    const [ventureReport, setVentureReport] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Simulate fetching data for both reports
        setTimeout(() => {
            setReport(Math.random() > 0.5 ? placeholderReport : noIssuesReport);
            setVentureReport(placeholderVentureReport);
            setLoading(false);
        }, 1800);
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
                ) : report ? (
                    <Box>
                        <Typography variant="subtitle1" gutterBottom>
                            {report.overall_summary}
                        </Typography>
                        <Divider sx={{ my: 1 }} />
                        <Typography variant="h6" sx={{ mt: 2, fontSize: '1rem', color: hasIssues ? 'error.main' : 'text.primary' }}>
                            {hasIssues ? "Potential Issues Identified" : "No Cost Anomalies Detected"}
                        </Typography>
                        <List dense>
                            {hasIssues ? (
                                report.potential_issues.map((issue: any, index: number) => (
                                    <ListItem key={index}>
                                        <ListItemText
                                            primary={`${issue.type}: ${issue.service}`}
                                            secondary={issue.details}
                                        />
                                    </ListItem>
                                ))
                            ) : (
                                <ListItem>
                                    <ListItemText primary={report.recommendation} />
                                </ListItem>
                            )}
                        </List>
                        
                        {/* --- NEW: Venture P&L Section --- */}
                        {ventureReport && (
                            <>
                                <Divider sx={{ my: 2 }} />
                                <Typography variant="h6" sx={{ mt: 2, fontSize: '1rem' }}>
                                    Autonomous Ventures P&L
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    Total Ventures: {ventureReport.total_ventures}
                                </Typography>
                                <Typography variant="body2" color="success.main">
                                    Total Revenue: ${ventureReport.total_revenue_usd.toFixed(2)}
                                </Typography>
                                <Typography variant="body2" color="error.main">
                                    Est. Costs: ${ventureReport.total_cost_usd.toFixed(2)}
                                </Typography>
                                <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                                    Net Profit: ${ventureReport.net_profit_usd.toFixed(2)}
                                </Typography>
                            </>
                        )}
                        {/* --- End Venture P&L Section --- */}
                    </Box>
                ) : (
                     <Typography variant="body1" color="text.secondary">
                        FinOps report is not available.
                    </Typography>
                )}
            </CardContent>
        </Card>
    );
} 