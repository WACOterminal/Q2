import React, { useEffect, useRef, useState } from 'react';
import { Box, CircularProgress, Paper, Typography, Button } from '@mui/material';
import { workflowBridge, WorldState } from './O3DEWorkflowBridge';
import { triggerWorkflow } from '../../services/managerAPI'; // Assumes this function exists
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

const O3DE_WEBSOCKET_URL = "ws://localhost:8004/v1/observability/ws"; // This should come from config

export function O3DEWorkflowVisualizer() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isEngineInitialized, setIsEngineInitialized] = useState(false);
    const [worldState, setWorldState] = useState<WorldState | null>(null);
    const [simulationStatus, setSimulationStatus] = useState<string | null>(null);
    const simulationId = useRef<string | null>(null);

    useEffect(() => {
        // Connect the bridge to the backend WebSocket
        workflowBridge.connect(O3DE_WEBSOCKET_URL);

        // Subscribe to world state updates
        workflowBridge.onUpdate((newState) => {
            setWorldState(newState);
        });

        // Initialize the O3DE engine
        // In a real application, the O3DE Gem would expose a global function
        // to start the engine and attach it to a canvas.
        const o3deModule = (window as any).o3de; 
        if (o3deModule && canvasRef.current) {
            o3deModule.start({ canvas: canvasRef.current }).then(() => {
                setIsEngineInitialized(true);
                console.log("O3DE Engine Initialized.");
            });
        }

        return () => {
            workflowBridge.disconnect();
            // o3deModule.stop();
        };
    }, []);

    useEffect(() => {
        // Pass the updated world state to the O3DE engine
        if (isEngineInitialized && worldState) {
            const o3deModule = (window as any).o3de;
            if (o3deModule && o3deModule.System) {
                 // This calls the C++ function we implemented
                o3deModule.System.UpdateWorldState(JSON.stringify(worldState));
            }
        }
    }, [worldState, isEngineInitialized]);

    const handleRunSimulation = async () => {
        try {
            setSimulationStatus("STARTING");
            const result = await triggerWorkflow('wf_chaos_engineering_test');
            // This is a simplification. A real implementation would need to get
            // the simulation ID from the workflow's final result.
            // For now, we'll just log it.
            console.log("Chaos engineering workflow started, task_id:", result.task_id);
            setSimulationStatus("RUNNING");
        } catch (error) {
            console.error("Failed to start simulation workflow:", error);
            setSimulationStatus(`FAILED: ${(error as Error).message}`);
        }
    };

    return (
        <Paper elevation={3} sx={{ height: 'calc(100vh - 200px)', p: 2, position: 'relative' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h5" gutterBottom>
                    Ethereal Twin Simulation Environment
                </Typography>
                <Button 
                    variant="contained" 
                    startIcon={<PlayArrowIcon />} 
                    onClick={handleRunSimulation}
                    disabled={simulationStatus === 'RUNNING'}
                >
                    Run Chaos Test
                </Button>
            </Box>

            {simulationStatus && <Typography>Status: {simulationStatus}</Typography>}
            
            {!isEngineInitialized && (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                    <CircularProgress />
                    <Typography sx={{ ml: 2 }}>Loading 3D Engine...</Typography>
                </Box>
            )}
            <canvas 
                ref={canvasRef} 
                style={{ 
                    width: '100%', 
                    height: '100%', 
                    display: isEngineInitialized ? 'block' : 'none' 
                }} 
            />
        </Paper>
    );
} 