import React, { useState, useContext } from 'react';
import { useParams } from 'react-router-dom';
import { 
    Box, 
    Typography, 
    Button, 
    ButtonGroup, 
    Chip, 
    Switch, 
    FormControlLabel, 
    Alert,
    Snackbar
} from '@mui/material';
import { Link } from 'react-router-dom';
import { ThreeD, TwoD, ViewInAr } from '@mui/icons-material';
import WorkflowVisualizer from '../components/WorkflowVisualizer/WorkflowVisualizer';
import O3DEWorkflowVisualizer from '../components/O3DE/O3DEWorkflowVisualizer';
import { AuthContext } from '../AuthContext';

export const WorkflowDetailPage: React.FC = () => {
    const { workflowId } = useParams<{ workflowId: string }>();
    const authContext = useContext(AuthContext);
    const [use3D, setUse3D] = useState(false);
    const [enableCollaboration, setEnableCollaboration] = useState(false);
    const [showNotification, setShowNotification] = useState(false);
    const [notificationMessage, setNotificationMessage] = useState('');

    if (!workflowId) {
        return <div>Workflow ID not found.</div>;
    }

    const handleVisualizationModeChange = (mode: 'legacy' | '3d' | 'vr') => {
        if (mode === 'legacy') {
            setUse3D(false);
        } else if (mode === '3d') {
            setUse3D(true);
        } else if (mode === 'vr') {
            setUse3D(true);
            setNotificationMessage('VR mode enabled! Put on your VR headset.');
            setShowNotification(true);
        }
    };

    const handleCollaborationToggle = () => {
        setEnableCollaboration(!enableCollaboration);
        setNotificationMessage(
            enableCollaboration 
                ? 'Collaboration disabled' 
                : 'Collaboration enabled! Share the session ID with others.'
        );
        setShowNotification(true);
    };

    const handleError = (error: Error) => {
        console.error('Workflow visualization error:', error);
        setNotificationMessage(`Error: ${error.message}`);
        setShowNotification(true);
    };

    const handleCollaborationUserJoined = (user: any) => {
        setNotificationMessage(`${user.displayName} joined the session`);
        setShowNotification(true);
    };

    const handleCollaborationUserLeft = (userId: string) => {
        setNotificationMessage(`User left the session`);
        setShowNotification(true);
    };

    return (
        <Box sx={{ p: 3 }}>
            {/* Header */}
            <Box sx={{ mb: 3 }}>
                <Link to="/workflows">&larr; Back to Workflows</Link>
                <Typography variant="h4" sx={{ mt: 2, mb: 1 }}>
                    Workflow Details
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    Workflow ID: {workflowId}
                </Typography>
            </Box>

            {/* Visualization Controls */}
            <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                <ButtonGroup variant="outlined" size="small">
                    <Button
                        startIcon={<TwoD />}
                        onClick={() => handleVisualizationModeChange('legacy')}
                        variant={!use3D ? 'contained' : 'outlined'}
                    >
                        Legacy 2D
                    </Button>
                    <Button
                        startIcon={<ThreeD />}
                        onClick={() => handleVisualizationModeChange('3d')}
                        variant={use3D ? 'contained' : 'outlined'}
                    >
                        3D View
                    </Button>
                    <Button
                        startIcon={<ViewInAr />}
                        onClick={() => handleVisualizationModeChange('vr')}
                        disabled={!use3D}
                    >
                        VR Mode
                    </Button>
                </ButtonGroup>

                <FormControlLabel
                    control={
                        <Switch
                            checked={enableCollaboration}
                            onChange={handleCollaborationToggle}
                            disabled={!use3D}
                        />
                    }
                    label="Enable Collaboration"
                />

                {use3D && (
                    <Chip
                        label="Enhanced 3D Visualization"
                        color="primary"
                        size="small"
                        variant="outlined"
                    />
                )}

                {enableCollaboration && (
                    <Chip
                        label="Collaboration Active"
                        color="success"
                        size="small"
                        variant="outlined"
                    />
                )}
            </Box>

            {/* Feature Notice */}
            {use3D && (
                <Alert severity="info" sx={{ mb: 3 }}>
                    <Typography variant="subtitle2">
                        🎮 3D Visualization Features:
                    </Typography>
                    <Typography variant="body2">
                        • Interactive 3D workflow navigation with mouse/touch controls
                        • Real-time collaboration with voice chat and spatial audio
                        • AI-powered layout optimization for complex workflows
                        • Immersive VR/AR support for enhanced visualization
                        • Advanced performance monitoring and adaptive quality
                    </Typography>
                </Alert>
            )}

            {/* Visualization Component */}
            <Box sx={{ height: '70vh' }}>
                {use3D ? (
                    <O3DEWorkflowVisualizer
                        workflowId={workflowId}
                        userId={authContext?.keycloak?.tokenParsed?.sub || 'anonymous'}
                        enableCollaboration={enableCollaboration}
                        enableVR={true}
                        enableAR={true}
                        onError={handleError}
                        onCollaborationUserJoined={handleCollaborationUserJoined}
                        onCollaborationUserLeft={handleCollaborationUserLeft}
                    />
                ) : (
                    <WorkflowVisualizer workflowId={workflowId} />
                )}
            </Box>

            {/* Notification Snackbar */}
            <Snackbar
                open={showNotification}
                autoHideDuration={4000}
                onClose={() => setShowNotification(false)}
                message={notificationMessage}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
            />
        </Box>
    );
}; 