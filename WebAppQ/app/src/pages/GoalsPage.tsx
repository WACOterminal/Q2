
import React, { useState, useEffect } from 'react';
import { Box, Typography, List, ListItem, ListItemText, CircularProgress, Alert, Paper, Button, Dialog, DialogTitle, DialogContent, TextField, DialogActions, ListItemButton } from '@mui/material';
import { Link } from 'react-router-dom';
import { listGoals, createGoal } from '../../services/managerAPI'; // Import createGoal

// Define a type for the Goal object we expect from the API
interface Goal {
    id: number;
    subject: string;
    status: string;
}

const GoalsPage: React.FC = () => {
    const [goals, setGoals] = useState<Goal[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const [openCreateDialog, setOpenCreateDialog] = useState(false);
    const [newGoalSubject, setNewGoalSubject] = useState('');

    const fetchGoals = async () => {
        try {
            setLoading(true);
            const data = await listGoals();
            setGoals(data);
            setError(null);
        } catch (err: any) {
            setError(err.message || 'An unexpected error occurred.');
            console.error("Failed to fetch goals:", err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchGoals();
    }, []);

    const handleOpenCreateDialog = () => {
        setOpenCreateDialog(true);
    };

    const handleCloseCreateDialog = () => {
        setOpenCreateDialog(false);
        setNewGoalSubject('');
    };

    const handleCreateGoal = async () => {
        if (!newGoalSubject.trim()) {
            alert("Please enter a subject for the goal.");
            return;
        }
        try {
            // Assuming project ID 1 for now
            await createGoal({ subject: newGoalSubject, project_id: 1 });
            handleCloseCreateDialog();
            // Refresh the list to show the new goal
            await fetchGoals(); 
        } catch (err: any) {
            alert(`Failed to create goal: ${err.message}`);
        }
    };

    return (
        <Box sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h4" gutterBottom>
                    Project Goals
                </Typography>
                <Button variant="contained" onClick={handleOpenCreateDialog}>
                    Create Goal
                </Button>
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                This list is sourced directly from OpenProject work packages.
            </Typography>

            <Paper>
                {loading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                        <CircularProgress />
                    </Box>
                )}
                {error && <Alert severity="error">{error}</Alert>}
                {!loading && !error && (
                    <List>
                        {goals.map((goal) => (
                            <ListItemButton component={Link} to={`/goals/${goal.id}`} key={goal.id} divider>
                                <ListItemText 
                                    primary={goal.subject}
                                    secondary={`Status: ${goal.status}`} 
                                />
                            </ListItemButton>
                        ))}
                    </List>
                )}
            </Paper>

            <Dialog open={openCreateDialog} onClose={handleCloseCreateDialog} fullWidth>
                <DialogTitle>Create New Goal</DialogTitle>
                <DialogContent>
                    <TextField
                        autoFocus
                        margin="dense"
                        id="subject"
                        label="Goal Subject"
                        type="text"
                        fullWidth
                        variant="standard"
                        value={newGoalSubject}
                        onChange={(e) => setNewGoalSubject(e.target.value)}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseCreateDialog}>Cancel</Button>
                    <Button onClick={handleCreateGoal}>Create</Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default GoalsPage; 