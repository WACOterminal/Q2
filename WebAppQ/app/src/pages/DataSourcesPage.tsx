import React, { useState } from 'react';
import { Box, Typography, TextField, Button, Container, Paper, CircularProgress, Snackbar, Alert } from '@mui/material';
import { ingestDataSource } from '../services/managerAPI'; // This function will be created later

export function DataSourcesPage() {
    const [url, setUrl] = useState('');
    const [loading, setLoading] = useState(false);
    const [feedback, setFeedback] = useState<{ open: boolean, message: string, severity: 'success' | 'error' }>({
        open: false,
        message: '',
        severity: 'success',
    });

    const handleSubmit = async (event: React.FormEvent) => {
        event.preventDefault();
        setLoading(true);

        try {
            // We will define this API function in a later step
            await ingestDataSource({ url }); 
            setFeedback({ open: true, message: 'Ingestion request received! The new data will be available shortly.', severity: 'success' });
            setUrl('');
        } catch (error) {
            console.error("Failed to ingest data source:", error);
            setFeedback({ open: true, message: 'Failed to submit ingestion request. Please try again.', severity: 'error' });
        } finally {
            setLoading(false);
        }
    };

    const handleCloseSnackbar = () => {
        setFeedback({ ...feedback, open: false });
    };

    return (
        <Container maxWidth="md" sx={{ mt: 4 }}>
            <Paper sx={{ p: 4 }}>
                <Typography variant="h4" component="h1" gutterBottom>
                    Ingest New Data Source
                </Typography>
                <Typography variant="body1" sx={{ mb: 3 }}>
                    Enter a URL to a web page, and the system will automatically process and index its content, making it available for search and analysis.
                </Typography>
                <form onSubmit={handleSubmit}>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <TextField
                            label="URL"
                            variant="outlined"
                            fullWidth
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            required
                            disabled={loading}
                        />
                        <Box sx={{ position: 'relative' }}>
                            <Button
                                type="submit"
                                variant="contained"
                                size="large"
                                fullWidth
                                disabled={loading}
                            >
                                Ingest Data
                            </Button>
                            {loading && (
                                <CircularProgress
                                    size={24}
                                    sx={{
                                        position: 'absolute',
                                        top: '50%',
                                        left: '50%',
                                        marginTop: '-12px',
                                        marginLeft: '-12px',
                                    }}
                                />
                            )}
                        </Box>
                    </Box>
                </form>
            </Paper>
            <Snackbar
                open={feedback.open}
                autoHideDuration={6000}
                onClose={handleCloseSnackbar}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
            >
                <Alert onClose={handleCloseSnackbar} severity={feedback.severity} sx={{ width: '100%' }}>
                    {feedback.message}
                </Alert>
            </Snackbar>
        </Container>
    );
} 