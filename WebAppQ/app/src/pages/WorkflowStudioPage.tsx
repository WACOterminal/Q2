// WebAppQ/app/src/pages/WorkflowStudioPage.tsx
import React, { useState, useCallback } from 'react';
import { Container, Typography } from '@mui/material';
import { WorkflowBuilder } from '../components/Workflows/WorkflowBuilder';
import { WorkflowTutorial } from '../components/Workflows/WorkflowTutorial';
import { useLocation } from 'react-router-dom';

export const WorkflowStudioPage: React.FC = () => {
    const location = useLocation();
    const [runTutorial, setRunTutorial] = useState(location.state?.startTutorial || false);

    const handleJoyrideCallback = useCallback((data: any) => {
        const { status } = data;
        if (['finished', 'skipped'].includes(status)) {
            setRunTutorial(false);
        }
    }, []);

    return (
        <Container maxWidth="xl" sx={{ mt: 4, height: 'calc(100vh - 100px)' }}>
            <WorkflowTutorial run={runTutorial} callback={handleJoyrideCallback} />
            <Typography variant="h4" gutterBottom>
                Workflow Studio
            </Typography>
            <Typography paragraph color="text.secondary">
                Design, build, and manage your automated workflows. Drag nodes from the sidebar onto the canvas to get started.
            </Typography>
            <WorkflowBuilder />
        </Container>
    );
}; 