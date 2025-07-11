import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import TimelineIcon from '@mui/icons-material/Timeline';
import BoltIcon from '@mui/icons-material/Bolt';
import PsychologyIcon from '@mui/icons-material/Psychology';

const SpikeTrain = ({ spikes }: { spikes: any[] }) => {
    return (
        <Box height={100} bgcolor="#222" my={2} p={1} sx={{ overflow: 'hidden', position: 'relative' }}>
            {spikes.map((spike, i) => (
                <Box 
                    key={i} 
                    sx={{ 
                        position: 'absolute', 
                        bottom: `${Math.random() * 90}%`, 
                        left: `${(spike.time - (Date.now() - 5000)) / 50}%`,
                        width: '2px', 
                        height: '10px', 
                        bgcolor: 'primary.main',
                        opacity: spike.intensity,
                    }}
                />
            ))}
        </Box>
    );
};

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
    const [avgWeight, setAvgWeight] = useState(0.5);

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

    useEffect(() => {
        const interval = setInterval(() => {
            // Simulate STDP learning by slowly changing the average weight
            setAvgWeight(w => Math.min(1.0, Math.max(0.1, w + (Math.random() - 0.5) * 0.01)));
        }, 200);
        return () => clearInterval(interval);
    }, []);

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
                
                <SpikeTrain spikes={spikeTrain} />
                
                {anomaly && <Chip label={`Anomaly: ${anomaly.text}`} color="error" />}
                 
                <Box display="flex" justifyContent="space-between" alignItems="center" mt={2}>
                    <Box display="flex" alignItems="center">
                        <BoltIcon color="success" />
                        <Typography variant="body2" ml={1}>
                            SNN Energy Usage: 0.02 µJ/inf
                        </Typography>
                    </Box>
                    <Box display="flex" alignItems="center">
                        <PsychologyIcon color="secondary" />
                        <Typography variant="body2" ml={1}>
                            Avg. Synaptic Weight: {avgWeight.toFixed(3)}
                        </Typography>
                    </Box>
                </Box>
            </CardContent>
        </Card>
    );
} 