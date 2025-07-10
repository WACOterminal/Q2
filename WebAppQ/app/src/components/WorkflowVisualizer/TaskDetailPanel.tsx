
import React, { useState } from 'react';
import { Box, Typography, Tabs, Tab, Paper, Dialog, DialogTitle, DialogContent, DialogActions, Button } from '@mui/material';
import { TaskBlock, WorkflowTask } from '../../services/types'; // Assuming types are defined here
import LogViewer from './LogViewer';

interface TaskDetailPanelProps {
    task: TaskBlock | null;
    open: boolean;
    onClose: () => void;
}

const TaskDetailPanel: React.FC<TaskDetailPanelProps> = ({ task, open, onClose }) => {
    const [activeTab, setActiveTab] = useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setActiveTab(newValue);
    };

    const isWorkflowTask = (task: TaskBlock): task is WorkflowTask => {
        return task.type === 'task';
    }

    if (!task) {
        return null;
    }

    return (
        <Dialog open={open} onClose={onClose} fullWidth maxWidth="md">
            <DialogTitle>
                Task Details: {task.task_id}
            </DialogTitle>
            <DialogContent>
                <Paper square>
                    <Tabs value={activeTab} onChange={handleTabChange} indicatorColor="primary" textColor="primary">
                        <Tab label="Details" />
                        <Tab label="Logs" />
                    </Tabs>
                </Paper>
                <Box sx={{ p: 2, mt: 2 }}>
                    {activeTab === 0 && (
                        <Box>
                            <Typography variant="h6">Details</Typography>
                            <Typography><strong>Type:</strong> {task.type}</Typography>
                            <Typography><strong>Status:</strong> {task.status}</Typography>
                            {isWorkflowTask(task) && <Typography><strong>Agent:</strong> {task.agent_personality}</Typography>}
                            <Typography variant="h6" sx={{ mt: 2 }}>Prompt</Typography>
                            <Paper variant="outlined" sx={{ p: 2, mt: 1, whiteSpace: 'pre-wrap', maxHeight: '200px', overflow: 'auto' }}>
                                {isWorkflowTask(task) ? task.prompt : 'N/A for this block type'}
                            </Paper>
                            <Typography variant="h6" sx={{ mt: 2 }}>Result</Typography>
                             <Paper variant="outlined" sx={{ p: 2, mt: 1, whiteSpace: 'pre-wrap', maxHeight: '200px', overflow: 'auto' }}>
                                {isWorkflowTask(task) ? task.result || 'No result yet.' : 'N/A for this block type'}
                            </Paper>
                        </Box>
                    )}
                    {activeTab === 1 && (
                        <Box>
                            <Typography variant="h6">Real-Time Logs</Typography>
                            <LogViewer taskId={task.task_id} />
                        </Box>
                    )}
                </Box>
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Close</Button>
            </DialogActions>
        </Dialog>
    );
};

export default TaskDetailPanel; 