import React, { useEffect, useState } from 'react';
import { Box, Typography, Paper, Grid, CircularProgress, Alert, Card, CardContent, Button } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Link } from 'react-router-dom';

interface MLMetrics {
  federated_learning: any;
  automl: any;
  reinforcement_learning: any;
  multimodal_ai: any;
  timestamp: string;
}

const MLDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<MLMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMLMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch('/v1/ml/integration/ml-capabilities-status'); // Assuming managerQ is proxied or directly accessible
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: MLMetrics = await response.json();
      setMetrics(data);
    } catch (e: any) {
      setError(e.message);
      console.error("Failed to fetch ML metrics:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMLMetrics();
    const interval = setInterval(fetchMLMetrics, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const renderCard = (title: string, content: React.ReactNode) => (
    <Grid item xs={12} sm={6} md={4} lg={3}>
      <Card raised sx={{ height: '100%' }}>
        <CardContent>
          <Typography variant="h6" component="div" gutterBottom>
            {title}
          </Typography>
          {content}
        </CardContent>
      </Card>
    </Grid>
  );

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ ml: 2 }}>Loading ML Dashboard...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">
          Error loading ML metrics: {error}. Please ensure the ManagerQ service is running and accessible.
          <Button onClick={fetchMLMetrics} sx={{ ml: 2 }} variant="outlined">Retry</Button>
        </Alert>
      </Box>
    );
  }

  if (!metrics) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="info">No ML metrics data available.</Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        MLOps Dashboard
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" gutterBottom>
        Real-time overview of Federated Learning, AutoML, Reinforcement Learning, and Multi-modal AI capabilities.
        (Last updated: {new Date(metrics.timestamp).toLocaleTimeString()})
      </Typography>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        {/* Federated Learning Metrics */}
        {renderCard(
          "Federated Learning",
          <>
            <Typography variant="body1">Total Rounds: {metrics.federated_learning.orchestrator_metrics?.total_rounds ?? 'N/A'}</Typography>
            <Typography variant="body1">Successful Rounds: {metrics.federated_learning.orchestrator_metrics?.successful_rounds ?? 'N/A'}</Typography>
            <Typography variant="body1">Avg Accuracy: {(metrics.federated_learning.orchestrator_metrics?.average_model_accuracy * 100).toFixed(2) ?? 'N/A'}%</Typography>
            <Typography variant="body1">Active Sessions: {metrics.federated_learning.active_sessions ?? 'N/A'}</Typography>
          </>
        )}

        {/* AutoML Metrics */}
        {renderCard(
          "AutoML Service",
          <>
            <Typography variant="body1">Experiments Completed: {metrics.automl.service_metrics?.experiments_completed ?? 'N/A'}</Typography>
            <Typography variant="body1">Models Trained: {metrics.automl.service_metrics?.models_trained ?? 'N/A'}</Typography>
            <Typography variant="body1">Best Accuracy: {(metrics.automl.service_metrics?.best_accuracy_achieved * 100).toFixed(2) ?? 'N/A'}%</Typography>
            <Typography variant="body1">Active Experiments: {metrics.automl.active_experiments ?? 'N/A'}</Typography>
          </>
        )}

        {/* Reinforcement Learning Metrics */}
        {renderCard(
          "Reinforcement Learning",
          <>
            <Typography variant="body1">Agents Trained: {metrics.reinforcement_learning.service_metrics?.agents_trained ?? 'N/A'}</Typography>
            <Typography variant="body1">Total Episodes: {metrics.reinforcement_learning.service_metrics?.total_episodes ?? 'N/A'}</Typography>
            <Typography variant="body1">Average Reward: {metrics.reinforcement_learning.service_metrics?.average_reward?.toFixed(2) ?? 'N/A'}</Typography>
            <Typography variant="body1">Best Reward: {metrics.reinforcement_learning.service_metrics?.best_reward?.toFixed(2) ?? 'N/A'}</Typography>
          </>
        )}

        {/* Multi-modal AI Metrics */}
        {renderCard(
          "Multi-modal AI",
          <>
            <Typography variant="body1">Requests Processed: {metrics.multimodal_ai.service_metrics?.requests_processed ?? 'N/A'}</Typography>
            <Typography variant="body1">Images Processed: {metrics.multimodal_ai.service_metrics?.images_processed ?? 'N/A'}</Typography>
            <Typography variant="body1">Audio Processed: {metrics.multimodal_ai.service_metrics?.audio_processed ?? 'N/A'}</Typography>
            <Typography variant="body1">Text Processed: {metrics.multimodal_ai.service_metrics?.text_processed ?? 'N/A'}</Typography>
          </>
        )}
      </Grid>

      {/* Optional: Add charts for historical data if available */}
      <Box sx={{ mt: 4 }}>
        <Typography variant="h5" component="h2" gutterBottom>
          Historical Trends (Placeholder)
        </Typography>
        <Paper sx={{ p: 2, height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={[]} // This would be populated with historical data from Elasticsearch
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="value" stroke="#8884d8" />
            </LineChart>
          </ResponsiveContainer>
          <Typography variant="body2" align="center" color="text.secondary">
            Connect to Elasticsearch to visualize historical trends here.
          </Typography>
        </Paper>
      </Box>

      <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
        <Button
          variant="contained"
          color="primary"
          onClick={fetchMLMetrics}
        >
          Refresh Metrics
        </Button>
      </Box>
    </Box>
  );
};

export default MLDashboard; 