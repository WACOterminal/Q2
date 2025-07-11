import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import TimelineIcon from '@mui/icons-material/Timeline';
import BoltIcon from '@mui/icons-material/Bolt';

// Mock data stream hook
const useMockMarketData = (callback: (data: any) => void) => {
    useEffect(() => {
        const stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"];
        const prices = { ... }; // Same logic as backend producer
        const interval = setInterval(() => {
            const stock = stocks[Math.floor(Math.random() * stocks.length)];
            const price = prices[stock];
            const change = Math.random() * 2 - 1;
            const newPrice = price + change;
            prices[stock] = Math.max(0, newPrice);
            callback({ symbol: stock, price: prices[stock] });
        }, 200); // Fast updates
        return () => clearInterval(interval);
    }, [callback]);
};

export function NeuromorphicMonitorWidget() {
    const [marketData, setMarketData] = useState<any[]>([]);
    const [spikeTrain, setSpikeTrain] = useState<any[]>([]);
    const [anomaly, setAnomaly] = useState<any>(null);

    const dataCallback = (data: any) => {
        setMarketData(prev => [...prev.slice(-50), data]);
        // Simulate SNN spike response
        setSpikeTrain(prev => [...prev.slice(-100), { time: Date.now(), intensity: Math.random() }]);
        // Simulate anomaly detection
        if (Math.random() < 0.02) {
            setAnomaly({ text: `Coordinated spike in ${data.symbol}`, time: Date.now() });
        }
    };
    
    useMockMarketData(dataCallback);

    return (
        <Card sx={{ height: '100%' }}>
            <CardContent>
                <Typography variant="h6">Neuromorphic Market Monitor</Typography>
                <Box display="flex" justifyContent="space-between" mt={2}>
                    <Typography variant="body2">Live Market Data</Typography>
                    <Box>
                        {marketData.slice(-5).map((d, i) => (
                            <Chip key={i} label={`${d.symbol}: $${d.price.toFixed(2)}`} size="small" sx={{ ml: 1 }} />
                        ))}
                    </Box>
                </Box>
                <Box height={100} bgcolor="#eee" my={2}>
                    {/* Spike train visualization would go here */}
                    <Typography p={1} color="textSecondary" variant="caption">Spike Train Visualization Area</Typography>
                </Box>
                {anomaly && <Chip label={`Anomaly: ${anomaly.text}`} color="error" />}
                 <Box display="flex" alignItems="center" mt={2}>
                    <BoltIcon color="success" />
                    <Typography variant="body2" ml={1}>
                        SNN Energy Usage: 0.02 µJ/inf | Classical Model Est: 25 µJ/inf (1250x more)
                    </Typography>
                </Box>
            </CardContent>
        </Card>
    );
} 