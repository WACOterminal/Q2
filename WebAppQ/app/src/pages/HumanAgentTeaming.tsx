import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  Avatar,
  LinearProgress,
  Tab,
  Tabs,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemAvatar,
  Badge,
  Tooltip,
  IconButton,
  Rating,
  Divider,
  Alert,
  Snackbar,
  CircularProgress
} from '@mui/material';
import {
  People,
  SmartToy,
  Add,
  Message,
  Assignment,
  Feedback,
  TrendingUp,
  Settings,
  VideoCall,
  Send,
  AttachFile,
  MoreVert,
  CheckCircle,
  Warning,
  Error,
  Info,
  Group,
  Psychology,
  Timeline,
  Analytics,
  Security
} from '@mui/icons-material';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
} from 'recharts';

// Types
interface Team {
  team_id: string;
  team_name: string;
  objective: string;
  human_members: Array<{
    id: string;
    type: string;
    status: string;
    skills: Record<string, number>;
    trust_level: string;
  }>;
  ai_agents: Array<{
    id: string;
    type: string;
    status: string;
    skills: Record<string, number>;
    trust_level: string;
    performance_metrics: Record<string, number>;
  }>;
  team_role: string;
  interaction_mode: string;
  status: string;
  performance_metrics: Record<string, number>;
  trust_metrics: Record<string, number>;
  created_at: string;
  last_active: string;
}

interface Task {
  task_id: string;
  task_name: string;
  assigned_to: string;
  status: string;
  progress: number;
  estimated_completion?: string;
  human_oversight_required: boolean;
  ai_assistance_available: boolean;
  blockers: string[];
  recent_updates: Array<{
    timestamp: string;
    progress: number;
    status: string;
    notes: string;
  }>;
}

interface Message {
  message_id: string;
  team_id: string;
  sender_id: string;
  sender_type: string;
  message_type: string;
  content: Record<string, any>;
  recipients: string[];
  timestamp: string;
  responses: Array<any>;
}

interface CollaborationMetrics {
  team_id: string;
  performance_score: number;
  efficiency_rating: number;
  communication_quality: number;
  trust_level: number;
  human_satisfaction: number;
  ai_performance: number;
  task_completion_rate: number;
  collaboration_frequency: number;
  measurement_period: {
    start: string;
    end: string;
  };
}

const HumanAgentTeaming: React.FC = () => {
  // State management
  const [teams, setTeams] = useState<Team[]>([]);
  const [selectedTeam, setSelectedTeam] = useState<Team | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [metrics, setMetrics] = useState<CollaborationMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showTaskDialog, setShowTaskDialog] = useState(false);
  const [showFeedbackDialog, setShowFeedbackDialog] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  
  // Form states
  const [newTeam, setNewTeam] = useState({
    team_name: '',
    objective: '',
    human_members: [''],
    ai_agents: [''],
    team_role: 'collaborative',
    interaction_mode: 'hybrid',
    expected_duration: 3600,
    trust_requirements: {},
    skill_requirements: {}
  });
  
  const [newTask, setNewTask] = useState({
    task_name: '',
    task_description: '',
    task_complexity: 'moderate',
    assigned_to: '',
    human_oversight: false,
    ai_assistance: false,
    deadline: '',
    dependencies: [],
    success_criteria: {}
  });
  
  const [newMessage, setNewMessage] = useState({
    content: { text: '' },
    message_type: 'chat',
    recipients: [],
    priority: 'normal'
  });
  
  const [feedback, setFeedback] = useState({
    target_id: '',
    feedback_type: 'performance',
    rating: 5,
    comments: '',
    suggestions: []
  });
  
  // WebSocket connection
  const wsRef = useRef<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  
  // API Functions
  const fetchTeams = async () => {
    try {
      const response = await fetch('/api/human-agent-teaming/teams');
      const data = await response.json();
      setTeams(data);
    } catch (error) {
      console.error('Error fetching teams:', error);
      setSnackbar({ open: true, message: 'Failed to fetch teams', severity: 'error' });
    }
  };
  
  const fetchTeamTasks = async (teamId: string) => {
    try {
      const response = await fetch(`/api/human-agent-teaming/teams/${teamId}/tasks`);
      const data = await response.json();
      setTasks(data);
    } catch (error) {
      console.error('Error fetching tasks:', error);
      setSnackbar({ open: true, message: 'Failed to fetch tasks', severity: 'error' });
    }
  };
  
  const fetchTeamMessages = async (teamId: string) => {
    try {
      const response = await fetch(`/api/human-agent-teaming/teams/${teamId}/messages`);
      const data = await response.json();
      setMessages(data);
    } catch (error) {
      console.error('Error fetching messages:', error);
      setSnackbar({ open: true, message: 'Failed to fetch messages', severity: 'error' });
    }
  };
  
  const fetchTeamMetrics = async (teamId: string) => {
    try {
      const response = await fetch(`/api/human-agent-teaming/teams/${teamId}/metrics`);
      const data = await response.json();
      setMetrics(data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
      setSnackbar({ open: true, message: 'Failed to fetch metrics', severity: 'error' });
    }
  };
  
  const createTeam = async () => {
    try {
      const response = await fetch('/api/human-agent-teaming/teams/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newTeam)
      });
      
      if (response.ok) {
        const data = await response.json();
        setSnackbar({ open: true, message: 'Team created successfully', severity: 'success' });
        setShowCreateDialog(false);
        fetchTeams();
      } else {
        throw new Error('Failed to create team');
      }
    } catch (error) {
      console.error('Error creating team:', error);
      setSnackbar({ open: true, message: 'Failed to create team', severity: 'error' });
    }
  };
  
  const delegateTask = async () => {
    if (!selectedTeam) return;
    
    try {
      const response = await fetch(`/api/human-agent-teaming/teams/${selectedTeam.team_id}/tasks/delegate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...newTask, team_id: selectedTeam.team_id })
      });
      
      if (response.ok) {
        setSnackbar({ open: true, message: 'Task delegated successfully', severity: 'success' });
        setShowTaskDialog(false);
        fetchTeamTasks(selectedTeam.team_id);
      } else {
        throw new Error('Failed to delegate task');
      }
    } catch (error) {
      console.error('Error delegating task:', error);
      setSnackbar({ open: true, message: 'Failed to delegate task', severity: 'error' });
    }
  };
  
  const sendMessage = async () => {
    if (!selectedTeam || !newMessage.content.text) return;
    
    try {
      const response = await fetch(`/api/human-agent-teaming/teams/${selectedTeam.team_id}/messages/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...newMessage,
          team_id: selectedTeam.team_id,
          sender_id: 'current_user', // This would be the actual user ID
          sender_type: 'human'
        })
      });
      
      if (response.ok) {
        setNewMessage({ ...newMessage, content: { text: '' } });
        fetchTeamMessages(selectedTeam.team_id);
      } else {
        throw new Error('Failed to send message');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setSnackbar({ open: true, message: 'Failed to send message', severity: 'error' });
    }
  };
  
  const submitFeedback = async () => {
    if (!selectedTeam) return;
    
    try {
      const response = await fetch(`/api/human-agent-teaming/teams/${selectedTeam.team_id}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...feedback, team_id: selectedTeam.team_id })
      });
      
      if (response.ok) {
        setSnackbar({ open: true, message: 'Feedback submitted successfully', severity: 'success' });
        setShowFeedbackDialog(false);
        fetchTeamMetrics(selectedTeam.team_id);
      } else {
        throw new Error('Failed to submit feedback');
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      setSnackbar({ open: true, message: 'Failed to submit feedback', severity: 'error' });
    }
  };
  
  // WebSocket connection
  const connectWebSocket = (teamId: string) => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    const ws = new WebSocket(`ws://localhost:8000/api/human-agent-teaming/teams/${teamId}/collaborate`);
    
    ws.onopen = () => {
      setConnectionStatus('connected');
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('Received message:', data);
      
      if (data.message_type === 'chat') {
        setMessages(prev => [data, ...prev]);
      } else if (data.message_type === 'task_update') {
        fetchTeamTasks(teamId);
      }
    };
    
    ws.onclose = () => {
      setConnectionStatus('disconnected');
      console.log('WebSocket disconnected');
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };
    
    wsRef.current = ws;
  };
  
  // Effects
  useEffect(() => {
    fetchTeams();
    setLoading(false);
  }, []);
  
  useEffect(() => {
    if (selectedTeam) {
      fetchTeamTasks(selectedTeam.team_id);
      fetchTeamMessages(selectedTeam.team_id);
      fetchTeamMetrics(selectedTeam.team_id);
      connectWebSocket(selectedTeam.team_id);
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [selectedTeam]);
  
  // Helper functions
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'paused': return 'warning';
      case 'completed': return 'info';
      case 'failed': return 'error';
      default: return 'default';
    }
  };
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle />;
      case 'failed': return <Error />;
      case 'paused': return <Warning />;
      default: return <Info />;
    }
  };
  
  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };
  
  // Render functions
  const renderTeamCard = (team: Team) => (
    <Card 
      key={team.team_id} 
      sx={{ 
        mb: 2, 
        cursor: 'pointer',
        border: selectedTeam?.team_id === team.team_id ? '2px solid #1976d2' : '1px solid #e0e0e0'
      }}
      onClick={() => setSelectedTeam(team)}
    >
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">{team.team_name}</Typography>
          <Chip 
            label={team.status} 
            color={getStatusColor(team.status)}
            icon={getStatusIcon(team.status)}
          />
        </Box>
        <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
          {team.objective}
        </Typography>
        <Box display="flex" alignItems="center" sx={{ mt: 2 }}>
          <Badge badgeContent={team.human_members.length} color="primary">
            <People />
          </Badge>
          <Badge badgeContent={team.ai_agents.length} color="secondary" sx={{ ml: 2 }}>
            <SmartToy />
          </Badge>
          <Typography variant="caption" sx={{ ml: 2 }}>
            {team.interaction_mode} • {team.team_role}
          </Typography>
        </Box>
        <Box sx={{ mt: 2 }}>
          <Typography variant="caption">Trust Level</Typography>
          <LinearProgress 
            variant="determinate" 
            value={team.trust_metrics.overall_trust * 100} 
            sx={{ mt: 0.5 }}
          />
        </Box>
      </CardContent>
    </Card>
  );
  
  const renderTaskCard = (task: Task) => (
    <Card key={task.task_id} sx={{ mb: 2 }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">{task.task_name}</Typography>
          <Chip label={task.status} color={getStatusColor(task.status)} />
        </Box>
        <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
          Assigned to: {task.assigned_to}
        </Typography>
        <Box sx={{ mt: 2 }}>
          <Typography variant="caption">Progress: {task.progress}%</Typography>
          <LinearProgress variant="determinate" value={task.progress} sx={{ mt: 0.5 }} />
        </Box>
        {task.blockers.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color="error">Blockers:</Typography>
            {task.blockers.map((blocker, index) => (
              <Chip key={index} label={blocker} size="small" color="error" sx={{ mr: 1, mt: 0.5 }} />
            ))}
          </Box>
        )}
        <Box display="flex" alignItems="center" sx={{ mt: 2 }}>
          {task.human_oversight_required && (
            <Chip label="Human Oversight" size="small" color="warning" sx={{ mr: 1 }} />
          )}
          {task.ai_assistance_available && (
            <Chip label="AI Assistance" size="small" color="info" sx={{ mr: 1 }} />
          )}
        </Box>
      </CardContent>
    </Card>
  );
  
  const renderMessage = (message: Message) => (
    <ListItem key={message.message_id} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
      <Box display="flex" alignItems="center" sx={{ mb: 1 }}>
        <Avatar sx={{ mr: 1 }}>
          {message.sender_type === 'human' ? <People /> : <SmartToy />}
        </Avatar>
        <Typography variant="subtitle2">{message.sender_id}</Typography>
        <Typography variant="caption" sx={{ ml: 1 }}>
          {formatDateTime(message.timestamp)}
        </Typography>
      </Box>
      <Typography variant="body2" sx={{ ml: 5 }}>
        {message.content.text}
      </Typography>
    </ListItem>
  );
  
  const renderMetricsChart = () => {
    if (!metrics) return null;
    
    const chartData = [
      { name: 'Performance', value: metrics.performance_score * 100 },
      { name: 'Efficiency', value: metrics.efficiency_rating * 100 },
      { name: 'Communication', value: metrics.communication_quality * 100 },
      { name: 'Trust', value: metrics.trust_level * 100 },
      { name: 'Satisfaction', value: metrics.human_satisfaction * 100 },
      { name: 'AI Performance', value: metrics.ai_performance * 100 }
    ];
    
    return (
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={chartData}>
          <PolarGrid />
          <PolarAngleAxis dataKey="name" />
          <PolarRadiusAxis domain={[0, 100]} />
          <Radar name="Metrics" dataKey="value" stroke="#1976d2" fill="#1976d2" fillOpacity={0.3} />
        </RadarChart>
      </ResponsiveContainer>
    );
  };
  
  const renderCreateTeamDialog = () => (
    <Dialog open={showCreateDialog} onClose={() => setShowCreateDialog(false)} maxWidth="md" fullWidth>
      <DialogTitle>Create New Team</DialogTitle>
      <DialogContent>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Team Name"
              value={newTeam.team_name}
              onChange={(e) => setNewTeam({ ...newTeam, team_name: e.target.value })}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Objective"
              multiline
              rows={3}
              value={newTeam.objective}
              onChange={(e) => setNewTeam({ ...newTeam, objective: e.target.value })}
            />
          </Grid>
          <Grid item xs={6}>
            <FormControl fullWidth>
              <InputLabel>Team Role</InputLabel>
              <Select
                value={newTeam.team_role}
                onChange={(e) => setNewTeam({ ...newTeam, team_role: e.target.value })}
              >
                <MenuItem value="collaborative">Collaborative</MenuItem>
                <MenuItem value="human_lead">Human Lead</MenuItem>
                <MenuItem value="ai_lead">AI Lead</MenuItem>
                <MenuItem value="human_supervisor">Human Supervisor</MenuItem>
                <MenuItem value="ai_assistant">AI Assistant</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6}>
            <FormControl fullWidth>
              <InputLabel>Interaction Mode</InputLabel>
              <Select
                value={newTeam.interaction_mode}
                onChange={(e) => setNewTeam({ ...newTeam, interaction_mode: e.target.value })}
              >
                <MenuItem value="synchronous">Synchronous</MenuItem>
                <MenuItem value="asynchronous">Asynchronous</MenuItem>
                <MenuItem value="hybrid">Hybrid</MenuItem>
                <MenuItem value="autonomous">Autonomous</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Human Members (comma-separated)"
              value={newTeam.human_members.join(', ')}
              onChange={(e) => setNewTeam({ ...newTeam, human_members: e.target.value.split(', ') })}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="AI Agents (comma-separated)"
              value={newTeam.ai_agents.join(', ')}
              onChange={(e) => setNewTeam({ ...newTeam, ai_agents: e.target.value.split(', ') })}
            />
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setShowCreateDialog(false)}>Cancel</Button>
        <Button onClick={createTeam} variant="contained">Create Team</Button>
      </DialogActions>
    </Dialog>
  );
  
  const renderTaskDialog = () => (
    <Dialog open={showTaskDialog} onClose={() => setShowTaskDialog(false)} maxWidth="md" fullWidth>
      <DialogTitle>Delegate Task</DialogTitle>
      <DialogContent>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Task Name"
              value={newTask.task_name}
              onChange={(e) => setNewTask({ ...newTask, task_name: e.target.value })}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Task Description"
              multiline
              rows={3}
              value={newTask.task_description}
              onChange={(e) => setNewTask({ ...newTask, task_description: e.target.value })}
            />
          </Grid>
          <Grid item xs={6}>
            <FormControl fullWidth>
              <InputLabel>Complexity</InputLabel>
              <Select
                value={newTask.task_complexity}
                onChange={(e) => setNewTask({ ...newTask, task_complexity: e.target.value })}
              >
                <MenuItem value="simple">Simple</MenuItem>
                <MenuItem value="moderate">Moderate</MenuItem>
                <MenuItem value="complex">Complex</MenuItem>
                <MenuItem value="expert">Expert</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6}>
            <FormControl fullWidth>
              <InputLabel>Assigned To</InputLabel>
              <Select
                value={newTask.assigned_to}
                onChange={(e) => setNewTask({ ...newTask, assigned_to: e.target.value })}
              >
                {selectedTeam?.human_members.map(member => (
                  <MenuItem key={member.id} value={member.id}>{member.id}</MenuItem>
                ))}
                {selectedTeam?.ai_agents.map(agent => (
                  <MenuItem key={agent.id} value={agent.id}>{agent.id}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={6}>
            <FormControl>
              <label>
                <input
                  type="checkbox"
                  checked={newTask.human_oversight}
                  onChange={(e) => setNewTask({ ...newTask, human_oversight: e.target.checked })}
                />
                Human Oversight Required
              </label>
            </FormControl>
          </Grid>
          <Grid item xs={6}>
            <FormControl>
              <label>
                <input
                  type="checkbox"
                  checked={newTask.ai_assistance}
                  onChange={(e) => setNewTask({ ...newTask, ai_assistance: e.target.checked })}
                />
                AI Assistance Available
              </label>
            </FormControl>
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setShowTaskDialog(false)}>Cancel</Button>
        <Button onClick={delegateTask} variant="contained">Delegate Task</Button>
      </DialogActions>
    </Dialog>
  );
  
  const renderFeedbackDialog = () => (
    <Dialog open={showFeedbackDialog} onClose={() => setShowFeedbackDialog(false)} maxWidth="sm" fullWidth>
      <DialogTitle>Submit Feedback</DialogTitle>
      <DialogContent>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12}>
            <FormControl fullWidth>
              <InputLabel>Target Member</InputLabel>
              <Select
                value={feedback.target_id}
                onChange={(e) => setFeedback({ ...feedback, target_id: e.target.value })}
              >
                {selectedTeam?.human_members.map(member => (
                  <MenuItem key={member.id} value={member.id}>{member.id}</MenuItem>
                ))}
                {selectedTeam?.ai_agents.map(agent => (
                  <MenuItem key={agent.id} value={agent.id}>{agent.id}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12}>
            <FormControl fullWidth>
              <InputLabel>Feedback Type</InputLabel>
              <Select
                value={feedback.feedback_type}
                onChange={(e) => setFeedback({ ...feedback, feedback_type: e.target.value })}
              >
                <MenuItem value="performance">Performance</MenuItem>
                <MenuItem value="collaboration">Collaboration</MenuItem>
                <MenuItem value="communication">Communication</MenuItem>
                <MenuItem value="overall">Overall</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12}>
            <Typography component="legend">Rating</Typography>
            <Rating
              name="rating"
              value={feedback.rating}
              onChange={(e, newValue) => setFeedback({ ...feedback, rating: newValue || 5 })}
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Comments"
              multiline
              rows={4}
              value={feedback.comments}
              onChange={(e) => setFeedback({ ...feedback, comments: e.target.value })}
            />
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setShowFeedbackDialog(false)}>Cancel</Button>
        <Button onClick={submitFeedback} variant="contained">Submit Feedback</Button>
      </DialogActions>
    </Dialog>
  );
  
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
        <CircularProgress />
      </Box>
    );
  }
  
  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" sx={{ mb: 3 }}>
        <Typography variant="h4">Human-Agent Teaming</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setShowCreateDialog(true)}
        >
          Create Team
        </Button>
      </Box>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Teams</Typography>
              <Box sx={{ maxHeight: 400, overflowY: 'auto' }}>
                {teams.map(renderTeamCard)}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={8}>
          {selectedTeam ? (
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                  <Typography variant="h6">{selectedTeam.team_name}</Typography>
                  <Box>
                    <Chip 
                      label={connectionStatus} 
                      color={connectionStatus === 'connected' ? 'success' : 'error'}
                      size="small"
                      sx={{ mr: 1 }}
                    />
                    <IconButton onClick={() => setShowTaskDialog(true)}>
                      <Assignment />
                    </IconButton>
                    <IconButton onClick={() => setShowFeedbackDialog(true)}>
                      <Feedback />
                    </IconButton>
                  </Box>
                </Box>
                
                <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
                  <Tab label="Overview" />
                  <Tab label="Tasks" />
                  <Tab label="Messages" />
                  <Tab label="Metrics" />
                </Tabs>
                
                {activeTab === 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                      {selectedTeam.objective}
                    </Typography>
                    
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="h6">Human Members</Typography>
                        <List>
                          {selectedTeam.human_members.map(member => (
                            <ListItem key={member.id}>
                              <ListItemAvatar>
                                <Avatar><People /></Avatar>
                              </ListItemAvatar>
                              <ListItemText 
                                primary={member.id} 
                                secondary={`Status: ${member.status}`}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Grid>
                      
                      <Grid item xs={6}>
                        <Typography variant="h6">AI Agents</Typography>
                        <List>
                          {selectedTeam.ai_agents.map(agent => (
                            <ListItem key={agent.id}>
                              <ListItemAvatar>
                                <Avatar><SmartToy /></Avatar>
                              </ListItemAvatar>
                              <ListItemText 
                                primary={agent.id} 
                                secondary={`Status: ${agent.status}`}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Grid>
                    </Grid>
                  </Box>
                )}
                
                {activeTab === 1 && (
                  <Box sx={{ mt: 2 }}>
                    <Box display="flex" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                      <Typography variant="h6">Tasks</Typography>
                      <Button
                        variant="contained"
                        size="small"
                        startIcon={<Add />}
                        onClick={() => setShowTaskDialog(true)}
                      >
                        Add Task
                      </Button>
                    </Box>
                    <Box sx={{ maxHeight: 400, overflowY: 'auto' }}>
                      {tasks.map(renderTaskCard)}
                    </Box>
                  </Box>
                )}
                
                {activeTab === 2 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="h6" sx={{ mb: 2 }}>Messages</Typography>
                    <Box sx={{ maxHeight: 300, overflowY: 'auto', mb: 2 }}>
                      <List>
                        {messages.map(renderMessage)}
                      </List>
                    </Box>
                    <Box display="flex" alignItems="center">
                      <TextField
                        fullWidth
                        variant="outlined"
                        placeholder="Type a message..."
                        value={newMessage.content.text}
                        onChange={(e) => setNewMessage({ ...newMessage, content: { text: e.target.value } })}
                        onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                      />
                      <IconButton onClick={sendMessage} sx={{ ml: 1 }}>
                        <Send />
                      </IconButton>
                    </Box>
                  </Box>
                )}
                
                {activeTab === 3 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="h6" sx={{ mb: 2 }}>Team Metrics</Typography>
                    {renderMetricsChart()}
                    
                    {metrics && (
                      <Grid container spacing={2} sx={{ mt: 2 }}>
                        <Grid item xs={6}>
                          <Typography variant="subtitle2">Performance Score</Typography>
                          <LinearProgress 
                            variant="determinate" 
                            value={metrics.performance_score * 100} 
                            sx={{ mt: 1 }}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="subtitle2">Trust Level</Typography>
                          <LinearProgress 
                            variant="determinate" 
                            value={metrics.trust_level * 100} 
                            sx={{ mt: 1 }}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="subtitle2">Task Completion Rate</Typography>
                          <LinearProgress 
                            variant="determinate" 
                            value={metrics.task_completion_rate * 100} 
                            sx={{ mt: 1 }}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="subtitle2">Human Satisfaction</Typography>
                          <LinearProgress 
                            variant="determinate" 
                            value={metrics.human_satisfaction * 100} 
                            sx={{ mt: 1 }}
                          />
                        </Grid>
                      </Grid>
                    )}
                  </Box>
                )}
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent>
                <Typography variant="h6" color="textSecondary" textAlign="center">
                  Select a team to view details
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
      
      {renderCreateTeamDialog()}
      {renderTaskDialog()}
      {renderFeedbackDialog()}
      
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert 
          onClose={() => setSnackbar({ ...snackbar, open: false })} 
          severity={snackbar.severity}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default HumanAgentTeaming; 