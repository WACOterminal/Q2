import React, { useState, useEffect, useRef } from 'react';
import { Box, TextField, Button, Paper, Typography, CircularProgress, Alert, Switch, FormControlLabel } from '@mui/material';
import { sendChatMessage } from '../../services/h2mAPI'; // Assumes this function exists
import { SuggestionModal } from './SuggestionModal'; // A new component to create

// --- NEW: Co-Pilot Types ---
interface ProposedAction {
    tool_name: string;
    parameters: Record<string, any>;
}

interface CoPilotApprovalRequest {
    type: 'copilot_approval_request';
    conversation_id: string;
    thought: string;
    proposed_action: ProposedAction;
    reply_topic: string;
}

const COPILOT_WS_BASE_URL = "ws://localhost:8002/api/v1/copilot/ws"; // Should be from config

const Chat: React.FC = () => {
    const [messages, setMessages] = useState<any[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [conversationId, setConversationId] = useState<string | null>(null);
    const [isCoPilotMode, setIsCoPilotMode] = useState(true);
    const [copilotRequest, setCopilotRequest] = useState<CoPilotApprovalRequest | null>(null);
    const [isSuggestionModalOpen, setIsSuggestionModalOpen] = useState(false);
    
    const ws = useRef<WebSocket | null>(null);

    useEffect(() => {
        // Cleanup WebSocket on component unmount
        return () => {
            ws.current?.close();
        };
    }, []);
    
    const setupWebSocket = (convId: string) => {
        if (ws.current) ws.current.close();

        ws.current = new WebSocket(`${COPILOT_WS_BASE_URL}/${convId}`);
        ws.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'copilot_approval_request') {
                setCopilotRequest(data);
                setIsLoading(false);
            }
        };
    };

    const handleSend = async () => {
        if (!input.trim()) return;
        setIsLoading(true);

        const convId = conversationId || `conv_${Date.now()}`;
        if (!conversationId) {
            setConversationId(convId);
            if(isCoPilotMode) setupWebSocket(convId);
        }
        
        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput('');

        try {
            const response = await sendChatMessage(convId, input, isCoPilotMode);
            if (!isCoPilotMode) {
                setMessages(prev => [...prev, { role: 'assistant', content: response.answer }]);
                setIsLoading(false);
            }
        } catch (error) {
            setMessages(prev => [...prev, { role: 'system', content: `Error: ${(error as Error).message}` }]);
            setIsLoading(false);
        }
    };
    
    const handleCopilotResponse = (decision: 'approve' | 'deny' | 'suggest', suggestion: string = '') => {
        if (!ws.current || !copilotRequest) return;
        
        const response = {
            decision,
            suggestion,
            reply_topic: copilotRequest.reply_topic
        };
        
        ws.current.send(JSON.stringify(response));
        setCopilotRequest(null);
        setIsLoading(true); // Agent is now working again
    };

    return (
        <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', height: 'calc(100vh - 100px)' }}>
            <Paper sx={{ flexGrow: 1, p: 2, overflowY: 'auto', mb: 2 }}>
                {messages.map((msg, index) => (
                    <Box key={index} sx={{ mb: 1, textAlign: msg.role === 'user' ? 'right' : 'left' }}>
                        <Chip label={msg.content} color={msg.role === 'user' ? 'primary' : 'default'} />
                    </Box>
                ))}
                {copilotRequest && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                        <Typography variant="body1"><b>Agent's Thought:</b> {copilotRequest.thought}</Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}><b>Proposed Action:</b> {copilotRequest.proposed_action.tool_name}</Typography>
                        <Box sx={{ mt: 2 }}>
                            <Button variant="contained" color="success" onClick={() => handleCopilotResponse('approve')}>Approve</Button>
                            <Button variant="outlined" color="error" sx={{ mx: 1 }} onClick={() => handleCopilotResponse('deny')}>Deny</Button>
                            <Button variant="outlined" onClick={() => setIsSuggestionModalOpen(true)}>Suggest</Button>
                        </Box>
                    </Alert>
                )}
            </Paper>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <FormControlLabel control={<Switch checked={isCoPilotMode} onChange={(e) => setIsCoPilotMode(e.target.checked)} />} label="Co-Pilot Mode" />
                <TextField fullWidth value={input} onChange={(e) => setInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSend()} />
                <Button onClick={handleSend} disabled={isLoading || !!copilotRequest}>Send</Button>
            </Box>
            <SuggestionModal
                open={isSuggestionModalOpen}
                onClose={() => setIsSuggestionModalOpen(false)}
                onSubmit={(suggestion) => handleCopilotResponse('suggest', suggestion)}
            />
        </Box>
    );
};

export default Chat;
