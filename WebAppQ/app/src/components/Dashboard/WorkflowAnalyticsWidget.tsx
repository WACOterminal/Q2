// WebAppQ/app/src/components/Dashboard/WorkflowAnalyticsWidget.tsx
import React, { useEffect, useState } from 'react';
import { Paper, Typography } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { fetchWorkflowAnalytics, WorkflowAnalyticData } from '../../services/elasticsearch';

export const WorkflowAnalyticsWidget: React.FC = () => {
    const [data, setData] = useState<WorkflowAnalyticData[]>([]);

    useEffect(() => {
        const getData = async () => {
            const analyticsData = await fetchWorkflowAnalytics();
            // For this chart, we want to aggregate the totals per agent
            const aggregatedData = analyticsData.reduce((acc, curr) => {
                const existing = acc.find(item => item.agent_id === curr.agent_id);
                if (existing) {
                    existing.success_count += curr.success_count;
                    existing.failure_count += curr.failure_count;
                } else {
                    acc.push({ ...curr });
                }
                return acc;
            }, [] as WorkflowAnalyticData[]);
            setData(aggregatedData);
        };
        getData();
    }, []);

    return (
        <Paper sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 400 }}>
            <Typography variant="h6" gutterBottom>
                Workflow Analytics (Success vs. Failure)
            </Typography>
            <ResponsiveContainer>
                <BarChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="agent_id" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="success_count" fill="#82ca9d" name="Success" />
                    <Bar dataKey="failure_count" fill="#8884d8" name="Failure" />
                </BarChart>
            </ResponsiveContainer>
        </Paper>
    );
}; 