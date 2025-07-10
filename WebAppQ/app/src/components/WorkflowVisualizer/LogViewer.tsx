
import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography } from '@mui/material';
import { connectToTaskLogsSocket, disconnectFromLogsSocket } from '../../services/managerAPI';

interface LogMessage {
    level: string;
    message: string;
    timestamp: number;
    details?: any;
}

interface LogViewerProps {
    taskId: string;
}

const LogViewer: React.FC<LogViewerProps> = ({ taskId }) => {
    const [logs, setLogs] = useState<LogMessage[]>([]);
    const logContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleNewLog = (log: LogMessage) => {
            setLogs(prevLogs => [...prevLogs, log]);
        };

        connectToTaskLogsSocket(taskId, handleNewLog);

        return () => {
            disconnectFromLogsSocket();
        };
    }, [taskId]);

    useEffect(() => {
        // Auto-scroll to the bottom
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <Box 
            ref={logContainerRef}
            sx={{ 
                height: '400px', 
                overflowY: 'auto', 
                bgcolor: '#1e1e1e', 
                color: '#d4d4d4', 
                fontFamily: 'monospace',
                p: 2,
                borderRadius: 1
            }}
        >
            {logs.map((log, index) => (
                <Typography key={index} component="div" sx={{ whiteSpace: 'pre-wrap', mb: 1 }}>
                    <span style={{ color: log.level === 'ERROR' ? '#f48771' : (log.level === 'WARN' ? '#f9d775' : '#569cd6') }}>
                        [{new Date(log.timestamp).toLocaleTimeString()}] [{log.level}]
                    </span>
                    <span style={{ marginLeft: '10px' }}>{log.message}</span>
                    {log.details && (
                        <span style={{ marginLeft: '10px', color: '#6a9955' }}>
                            {JSON.stringify(log.details)}
                        </span>
                    )}
                </Typography>
            ))}
        </Box>
    );
};

export default LogViewer; 