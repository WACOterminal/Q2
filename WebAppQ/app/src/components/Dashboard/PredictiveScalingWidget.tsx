import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, CircularProgress, Box, Chip, Tooltip } from '@mui/material';
import { Timeline, TimelineItem, TimelineSeparator, TimelineConnector, TimelineContent, TimelineDot, timelineOppositeContent, TimelineOppositeContent } from '@mui/lab';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward';
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';

// Placeholder data
const placeholderEvents = [
    {
        time: new Date(Date.now() - 3600 * 1000 * 2).toISOString(),
        type: 'forecast',
        details: 'Load predicted to increase by 40% in the next hour.'
    },
    {
        time: new Date(Date.now() - 3600 * 1000 * 1.9).toISOString(),
        type: 'action',
        details: 'Scaled up QuantumPulse replicas from 3 to 4.',
        icon: 'up'
    },
    {
        time: new Date(Date.now() - 3600 * 1000 * 1).toISOString(),
        type: 'forecast',
        details: 'Load predicted to decrease by 30% after peak hours.'
    },
    {
        time: new Date(Date.now() - 3600 * 1000 * 0.5).toISOString(),
        type: 'action',
        details: 'Scaled down QuantumPulse replicas from 4 to 3.',
        icon: 'down'
    }
];

export function PredictiveScalingWidget() {
    const [events, setEvents] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        setTimeout(() => {
            setEvents(placeholderEvents.sort((a, b) => new Date(b.time).getTime() - new Date(a.time).getTime()));
            setLoading(false);
        }, 1200);
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
                ) : (
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