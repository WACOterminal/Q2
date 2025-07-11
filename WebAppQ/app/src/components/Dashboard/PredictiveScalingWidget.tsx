import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, CircularProgress, Box, Tooltip } from '@mui/material';
import { Timeline, TimelineItem, TimelineSeparator, TimelineConnector, TimelineContent, TimelineDot, timelineOppositeContent, TimelineOppositeContent } from '@mui/lab';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import { getPredictiveScalingReport } from '../../services/managerAPI'; // Assuming this exists

export function PredictiveScalingWidget() {
    const [events, setEvents] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchReport = async () => {
            try {
                setLoading(true);
                const data = await getPredictiveScalingReport();
                setEvents(data.sort((a: any, b: any) => new Date(b.time).getTime() - new Date(a.time).getTime()));
            } catch (err) {
                setError((err as Error).message);
            } finally {
                setLoading(false);
            }
        };
        fetchReport();
    }, []);

    const getIcon = (event: any) => {
        if (event.type === 'forecast') return <AutoAwesomeIcon />;
        if (event.icon === 'up') return <ArrowUpwardIcon sx={{ color: 'success.main' }} />;
        if (event.icon === 'down') return <ArrowDownwardIcon sx={{ color: 'info.main' }} />;
        return <AutoAwesomeIcon />;
    };

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Typography variant="h6" gutterBottom>
                    Predictive Autoscaler Activity
                </Typography>
                {loading ? (
                    <CircularProgress />
                ) : error ? (
                    <Typography color="error">Failed to load activity: {error}</Typography>
                ): (
                    <Timeline position="right" sx={{ p: 0 }}>
                        {events.map((event, index) => (
                            <TimelineItem key={index}>
                                <TimelineOppositeContent sx={{ flex: 0.3, px: 1 }} color="text.secondary">
                                    {new Date(event.time).toLocaleTimeString()}
                                </TimelineOppositeContent>
                                <TimelineSeparator>
                                    <Tooltip title={event.type === 'action' ? 'Scaling Action' : 'Load Forecast'}>
                                        <TimelineDot color={event.type === 'action' ? 'success' : 'primary'}>
                                            {getIcon(event)}
                                        </TimelineDot>
                                    </Tooltip>
                                    {index < events.length - 1 && <TimelineConnector />}
                                </TimelineSeparator>
                                <TimelineContent sx={{ px: 2 }}>{event.details}</TimelineContent>
                            </TimelineItem>
                        ))}
                    </Timeline>
                )}
            </CardContent>
        </Card>
    );
} 