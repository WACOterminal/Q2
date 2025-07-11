import React, { useState, useEffect, useContext } from 'react';
import { Snackbar, Alert, Button, Dialog, DialogTitle, DialogContent, DialogActions, Typography } from '@mui/material';
import { AuthContext } from '../../AuthContext';

const PROPOSALS_WS_BASE_URL = "ws://localhost:8002/api/v1/proposals/ws"; // Should come from config

interface Suggestion {
    suggestion_text: string;
    action_type: string;
    action_payload: any;
}

export function ProactiveSuggestionToast() {
    const [suggestion, setSuggestion] = useState<Suggestion | null>(null);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const authContext = useContext(AuthContext);

    useEffect(() => {
        if (!authContext?.user?.profile.sub) return;

        const userId = authContext.user.profile.sub;
        const ws = new WebSocket(`${PROPOSALS_WS_BASE_URL}/${userId}`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setSuggestion(data);
        };

        return () => ws.close();
    }, [authContext]);

    const handleAccept = () => {
        // Here you would trigger the action, e.g., by calling a managerQ API
        console.log("User accepted suggestion:", suggestion);
        setSuggestion(null);
        setIsModalOpen(false);
    };

    return (
        <>
            <Snackbar open={!!suggestion && !isModalOpen} autoHideDuration={null}>
                <Alert severity="info" action={<Button onClick={() => setIsModalOpen(true)}>View</Button>}>
                    The Q Platform has a suggestion for you.
                </Alert>
            </Snackbar>
            <Dialog open={isModalOpen} onClose={() => setIsModalOpen(false)}>
                <DialogTitle>Proactive Suggestion</DialogTitle>
                <DialogContent>
                    <Typography>{suggestion?.suggestion_text}</Typography>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setIsModalOpen(false)}>Dismiss</Button>
                    <Button onClick={handleAccept} variant="contained">Accept</Button>
                </DialogActions>
            </Dialog>
        </>
    );
} 