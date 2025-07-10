// WebAppQ/app/src/pages/GettingStartedPage.tsx
import React from 'react';
import { Container, Typography, Button, Paper, Box } from '@mui/material';
import { RocketLaunch } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

export const GettingStartedPage: React.FC = () => {
    const navigate = useNavigate();

    const startTutorial = () => {
        navigate('/workflow-studio', { state: { startTutorial: true } });
    };

    return (
        <Container maxWidth="md" sx={{ mt: 4 }}>
            <Paper elevation={3} sx={{ p: 4 }}>
                <Box sx={{ textAlign: 'center', mb: 3 }}>
                    <Typography variant="h3" gutterBottom>
                        Welcome to the Q Platform!
                    </Typography>
                    <Typography variant="h6" color="text.secondary">
                        Your intelligent automation and orchestration studio.
                    </Typography>
                </Box>
                <Typography paragraph>
                    The Q Platform provides a powerful suite of tools to design, build, and manage complex automated workflows. Whether you're analyzing data, managing infrastructure, or integrating with external systems, our autonomous agents are here to help.
                </Typography>
                <Typography paragraph>
                    Ready to get started? Our interactive tutorial will guide you through the process of creating your first automated workflow in just a few minutes.
                </Typography>
                <Box sx={{ textAlign: 'center', mt: 4 }}>
                    <Button
                        variant="contained"
                        size="large"
                        startIcon={<RocketLaunch />}
                        onClick={startTutorial}
                    >
                        Start Interactive Tutorial
                    </Button>
                </Box>
            </Paper>
        </Container>
    );
}; 