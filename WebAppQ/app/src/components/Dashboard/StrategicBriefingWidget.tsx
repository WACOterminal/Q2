import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, CircularProgress, Box, List, ListItem, ListItemText, Divider, Paper } from '@mui/material';
import InsightsIcon from '@mui/icons-material/Insights';

// Placeholder data
const placeholderBriefing = {
    report_id: "briefing-q3-2024",
    title: "Quarterly Strategic Briefing",
    timestamp: new Date().toISOString(),
    insights: [
        {
            id: "insight-1",
            title: "High-Cost Services Show High Stability",
            summary: "The 'VectorStoreQ' and 'QuantumPulse' services account for 55% of cloud spend but are linked to less than 5% of production incidents. This indicates they are stable, high-value components ripe for targeted cost optimization rather than reactive fixes.",
            evidence: ["FinOps Summary", "RCA Summary"]
        },
        {
            id: "insight-2",
            title: "Security Debt Correlates with Workflow Failures",
            summary: "A recent spike in 'Hardcoded Secret' vulnerabilities corresponds with a 3% dip in the overall workflow success rate, suggesting that poor security practices are introducing instability into automated processes.",
            evidence: ["Security Summary", "Platform KPIs"]
        }
    ]
};

export function StrategicBriefingWidget() {
    const [briefing, setBriefing] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Simulate fetching data from a workflow result endpoint
        setTimeout(() => {
            setBriefing(placeholderBriefing);
            setLoading(false);
        }, 3000);
    }, []);

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <InsightsIcon sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="h6">
                        Quarterly Strategic Briefing
                    </Typography>
                </Box>
                {loading ? (
                    <CircularProgress />
                ) : briefing ? (
                    <List>
                        {briefing.insights.map((insight: any, index: number) => (
                            <Paper key={insight.id} variant="outlined" sx={{ p: 2, mb: 2 }}>
                                <Typography variant="subtitle1" component="div" gutterBottom>
                                    {insight.title}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                    {insight.summary}
                                </Typography>
                            </Paper>
                        ))}
                    </List>
                ) : (
                     <Typography variant="body1" color="text.secondary">
                        Strategic briefing is not yet available.
                    </Typography>
                )}
            </CardContent>
        </Card>
    );
} 