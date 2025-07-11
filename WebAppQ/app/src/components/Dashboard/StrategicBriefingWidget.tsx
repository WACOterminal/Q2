import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, CircularProgress, Box, List, ListItem, ListItemText, Divider, Paper } from '@mui/material';
import InsightsIcon from '@mui/icons-material/Insights';
import { getStrategicBriefing } from '../../services/managerAPI'; // Assuming this exists

export function StrategicBriefingWidget() {
    const [briefing, setBriefing] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchBriefing = async () => {
            try {
                setLoading(true);
                const data = await getStrategicBriefing();
                setBriefing(data);
            } catch (err) {
                setError((err as Error).message);
            } finally {
                setLoading(false);
            }
        };
        fetchBriefing();
    }, []);

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <InsightsIcon sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="h6">
                        {briefing?.title || 'Strategic Briefing'}
                    </Typography>
                </Box>
                {loading ? (
                    <CircularProgress />
                ) : error ? (
                    <Typography color="error">Failed to load briefing: {error}</Typography>
                ) : briefing ? (
                    <List>
                        {briefing.insights.map((insight: any) => (
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