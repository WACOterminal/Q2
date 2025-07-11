import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, CircularProgress, Box, Chip } from '@mui/material';
import { CheckCircle } from '@mui/icons-material';
import HubIcon from '@mui/icons-material/Hub';

// Placeholder data - in a real implementation, this would come from an API
const placeholderResult = {
    status: "SUCCESS",
    optimal_provider: {
        name: "anthropic-claude3-opus",
        cost_per_1k_tokens: 0.025,
        p90_latency_ms": 950
    },
    reason: "Quantum solver found an optimal balance between cost and latency, selecting provider 'anthropic-claude3-opus'.",
    timestamp: new Date().toISOString()
};

export function QuantumRoutingWidget() {
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Simulate fetching data from a workflow result endpoint
        setTimeout(() => {
            setResult(placeholderResult);
            setLoading(false);
        }, 2500);
    }, []);

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="h6" gutterBottom>
                        Quantum LLM Routing
                    </Typography>
                    {result && (
                        <Chip
                            icon={<HubIcon />}
                            label={`Optimal Provider: ${result.optimal_provider?.name}`}
                            color="primary"
                        />
                    )}
                </Box>
                {loading ? (
                    <CircularProgress />
                ) : result && result.status === 'SUCCESS' ? (
                    <Box>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                           Last optimization: {new Date(result.timestamp).toLocaleString()}
                        </Typography>
                        <Typography variant="body1">
                            {result.reason}
                        </Typography>
                    </Box>
                ) : (
                    <Typography variant="body1" color="error">
                        Optimization failed or is not available.
                    </Typography>
                )}
            </CardContent>
        </Card>
    );
} 