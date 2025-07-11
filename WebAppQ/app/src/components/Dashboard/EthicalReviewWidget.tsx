import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, Box, List, ListItem, ListItemText, Divider, Alert, AlertTitle } from '@mui/material';
import GppMaybeIcon from '@mui/icons-material/GppMaybe';

// Mock data and WebSocket connection
const useMockVetoEvents = (callback: (data: any) => void) => {
    useEffect(() => {
        const interval = setInterval(() => {
            if (Math.random() < 0.1) { // 10% chance per 5 seconds to get a new veto event
                const mockVetoEvent = {
                    workflow_id: `wf_${Math.random().toString(36).substr(2, 9)}`,
                    reason: "Proposed action violates Principle P002: Ensure System Stability & Security by attempting to merge an untested code change.",
                    timestamp: new Date().toISOString()
                };
                callback(mockVetoEvent);
            }
        }, 5000);
        return () => clearInterval(interval);
    }, [callback]);
};

export function EthicalReviewWidget() {
    const [vetoedWorkflows, setVetoedWorkflows] = useState<any[]>([]);

    const eventCallback = (data: any) => {
        setVetoedWorkflows(prev => [data, ...prev]);
    };

    useMockVetoEvents(eventCallback);

    if (vetoedWorkflows.length === 0) {
        return null; // Don't render anything if there are no vetoes
    }

    return (
        <Alert severity="error" icon={<GppMaybeIcon fontSize="inherit" />}>
            <AlertTitle>Ethical Review Alert - Action Required</AlertTitle>
            <Typography variant="body2" gutterBottom>
                The following workflows were automatically halted by the Guardian Agent Squad for violating the Platform Constitution. Manual review is required.
            </Typography>
            <List dense sx={{ maxHeight: 200, overflow: 'auto', bgcolor: 'background.paper', mt: 1 }}>
                {vetoedWorkflows.map((veto, index) => (
                    <ListItem key={index}>
                        <ListItemText
                            primary={`Workflow ID: ${veto.workflow_id}`}
                            secondary={`Reason: ${veto.reason}`}
                        />
                    </ListItem>
                ))}
            </List>
        </Alert>
    );
} 