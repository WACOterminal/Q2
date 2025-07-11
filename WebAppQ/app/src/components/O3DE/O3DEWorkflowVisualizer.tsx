import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { 
  Box, 
  Paper, 
  Button, 
  ButtonGroup, 
  IconButton, 
  Slider, 
  Typography, 
  Tooltip, 
  Switch, 
  FormControlLabel, 
  Select, 
  MenuItem, 
  FormControl, 
  InputLabel,
  Chip,
  Alert,
  CircularProgress,
  Drawer,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  Badge,
  Snackbar
} from '@mui/material';
import {
  ThreeD,
  TwoD,
  VolumeUp,
  VolumeOff,
  Settings,
  Fullscreen,
  FullscreenExit,
  ViewInAr,
  ViewModule,
  ZoomIn,
  ZoomOut,
  RotateLeft,
  RotateRight,
  Refresh,
  GridView,
  AccountCircle,
  Mic,
  MicOff,
  Share,
  Help,
  Accessibility,
  Speed,
  Visibility,
  VisibilityOff,
  ChatBubble,
  Group,
  AutoAwesome,
  Analytics
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { Workflow, TaskBlock } from '../../services/types';
import { getWorkflow, approveWorkflowTask, connectToDashboardSocket, disconnectFromDashboardSocket } from '../../services/managerAPI';
import { O3DEWorkflowBridge, VisualizationMode, QualityLevel, UserPresence, PerformanceMetrics, Vector3 } from './O3DEWorkflowBridge';
import TaskDetailPanel from '../WorkflowVisualizer/TaskDetailPanel';

// Styled components
const VisualizerContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  height: '100%',
  backgroundColor: theme.palette.background.default,
  overflow: 'hidden',
  display: 'flex',
  flexDirection: 'column'
}));

const CanvasContainer = styled(Box)(({ theme }) => ({
  position: 'relative',
  flex: 1,
  backgroundColor: '#000',
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden',
  '& canvas': {
    width: '100%',
    height: '100%',
    display: 'block'
  }
}));

const ControlPanel = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  left: theme.spacing(2),
  zIndex: 1000,
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1),
  '& > *': {
    backgroundColor: theme.palette.background.paper,
    backdropFilter: 'blur(10px)',
    borderRadius: theme.shape.borderRadius,
    boxShadow: theme.shadows[3]
  }
}));

const CollaborationPanel = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: theme.spacing(2),
  right: theme.spacing(2),
  zIndex: 1000,
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1),
  '& > *': {
    backgroundColor: theme.palette.background.paper,
    backdropFilter: 'blur(10px)',
    borderRadius: theme.shape.borderRadius,
    boxShadow: theme.shadows[3]
  }
}));

const PerformancePanel = styled(Box)(({ theme }) => ({
  position: 'absolute',
  bottom: theme.spacing(2),
  left: theme.spacing(2),
  zIndex: 1000,
  backgroundColor: theme.palette.background.paper,
  backdropFilter: 'blur(10px)',
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[3],
  padding: theme.spacing(1),
  minWidth: 200
}));

const StatusBar = styled(Box)(({ theme }) => ({
  position: 'absolute',
  bottom: 0,
  left: 0,
  right: 0,
  height: 32,
  backgroundColor: theme.palette.background.paper,
  borderTop: `1px solid ${theme.palette.divider}`,
  display: 'flex',
  alignItems: 'center',
  paddingLeft: theme.spacing(2),
  paddingRight: theme.spacing(2),
  zIndex: 1000
}));

const LoadingOverlay = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.7)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  zIndex: 2000,
  color: theme.palette.common.white
}));

// Interface definitions
interface O3DEWorkflowVisualizerProps {
  workflowId: string;
  userId?: string;
  sessionId?: string;
  enableCollaboration?: boolean;
  enableVR?: boolean;
  enableAR?: boolean;
  defaultMode?: VisualizationMode;
  defaultQuality?: QualityLevel;
  onNodeClick?: (nodeId: string) => void;
  onNodeDoubleClick?: (nodeId: string) => void;
  onEdgeClick?: (edgeId: string) => void;
  onCollaborationUserJoined?: (user: UserPresence) => void;
  onCollaborationUserLeft?: (userId: string) => void;
  onError?: (error: Error) => void;
}

interface CollaborationState {
  enabled: boolean;
  sessionId: string;
  users: UserPresence[];
  voiceEnabled: boolean;
  chatEnabled: boolean;
}

interface VisualizationState {
  mode: VisualizationMode;
  quality: QualityLevel;
  showGrid: boolean;
  showLabels: boolean;
  showAnimations: boolean;
  showParticles: boolean;
  enableSpatialAudio: boolean;
  isFullscreen: boolean;
}

interface PerformanceState {
  metrics: PerformanceMetrics;
  showMetrics: boolean;
  autoOptimize: boolean;
  targetFPS: number;
}

const O3DEWorkflowVisualizer: React.FC<O3DEWorkflowVisualizerProps> = ({
  workflowId,
  userId = 'user_' + Math.random().toString(36).substr(2, 9),
  sessionId = '',
  enableCollaboration = false,
  enableVR = false,
  enableAR = false,
  defaultMode = VisualizationMode.ThreeD,
  defaultQuality = QualityLevel.Medium,
  onNodeClick,
  onNodeDoubleClick,
  onEdgeClick,
  onCollaborationUserJoined,
  onCollaborationUserLeft,
  onError
}) => {
  // State management
  const [workflow, setWorkflow] = useState<Workflow | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTask, setSelectedTask] = useState<TaskBlock | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [collaborationOpen, setCollaborationOpen] = useState(false);
  const [showNotification, setShowNotification] = useState(false);
  const [notificationMessage, setNotificationMessage] = useState('');

  // Visualization state
  const [visualizationState, setVisualizationState] = useState<VisualizationState>({
    mode: defaultMode,
    quality: defaultQuality,
    showGrid: true,
    showLabels: true,
    showAnimations: true,
    showParticles: true,
    enableSpatialAudio: true,
    isFullscreen: false
  });

  // Collaboration state
  const [collaborationState, setCollaborationState] = useState<CollaborationState>({
    enabled: enableCollaboration,
    sessionId: sessionId,
    users: [],
    voiceEnabled: false,
    chatEnabled: false
  });

  // Performance state
  const [performanceState, setPerformanceState] = useState<PerformanceState>({
    metrics: {
      frameTime: 0,
      renderTime: 0,
      updateTime: 0,
      drawCalls: 0,
      vertices: 0,
      triangles: 0,
      memoryUsage: 0,
      gpuMemoryUsage: 0,
      activeNodes: 0,
      visibleNodes: 0,
      currentQuality: defaultQuality,
      cpuUsage: 0,
      gpuUsage: 0,
      networkLatency: 0,
      customMetrics: {}
    },
    showMetrics: false,
    autoOptimize: true,
    targetFPS: 60
  });

  // Refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bridgeRef = useRef<O3DEWorkflowBridge | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize O3DE bridge
  useEffect(() => {
    if (!canvasRef.current) return;

    const initializeBridge = async () => {
      try {
        setLoading(true);
        
        // Create unique canvas ID
        const canvasId = `o3de-canvas-${workflowId}`;
        canvasRef.current!.id = canvasId;

        // Initialize O3DE bridge
        const bridge = new O3DEWorkflowBridge(canvasId, userId);
        bridgeRef.current = bridge;

        // Set up event listeners
        bridge.on('initialized', () => {
          setIsInitialized(true);
          setLoading(false);
          console.log('O3DE bridge initialized');
        });

        bridge.on('error', (error: Error) => {
          setError(error.message);
          setLoading(false);
          if (onError) onError(error);
          console.error('O3DE bridge error:', error);
        });

        bridge.on('workflowCreated', (workflow: Workflow) => {
          console.log('Workflow visualization created:', workflow.workflow_id);
          setNotificationMessage('Workflow visualization created successfully');
          setShowNotification(true);
        });

        bridge.on('nodeStatusUpdated', ({ nodeId, status }: any) => {
          console.log('Node status updated:', nodeId, status);
          // Update workflow data
          if (workflow) {
            fetchWorkflow();
          }
        });

        bridge.on('userInteraction', (interaction: any) => {
          handleUserInteraction(interaction);
        });

        bridge.on('collaborativeUserAdded', (user: UserPresence) => {
          setCollaborationState(prev => ({
            ...prev,
            users: [...prev.users, user]
          }));
          if (onCollaborationUserJoined) onCollaborationUserJoined(user);
        });

        bridge.on('collaborativeUserRemoved', (userId: string) => {
          setCollaborationState(prev => ({
            ...prev,
            users: prev.users.filter(u => u.userId !== userId)
          }));
          if (onCollaborationUserLeft) onCollaborationUserLeft(userId);
        });

        bridge.on('performanceMetrics', (metrics: PerformanceMetrics) => {
          setPerformanceState(prev => ({ ...prev, metrics }));
        });

        bridge.on('qualityLevelChanged', (level: QualityLevel) => {
          setVisualizationState(prev => ({ ...prev, quality: level }));
        });

        bridge.on('visualizationModeChanged', (mode: VisualizationMode) => {
          setVisualizationState(prev => ({ ...prev, mode }));
        });

        // Initialize the bridge
        await bridge.initialize();

      } catch (error) {
        console.error('Failed to initialize O3DE bridge:', error);
        setError('Failed to initialize 3D visualization');
        setLoading(false);
        if (onError) onError(error as Error);
      }
    };

    initializeBridge();

    return () => {
      if (bridgeRef.current) {
        bridgeRef.current.dispose();
        bridgeRef.current = null;
      }
    };
  }, [workflowId, userId, onError]);

  // Fetch workflow data
  const fetchWorkflow = useCallback(async () => {
    if (!workflowId) return;

    try {
      const workflowData = await getWorkflow(workflowId);
      setWorkflow(workflowData);
      setError(null);

      // Create/update visualization
      if (bridgeRef.current && isInitialized) {
        if (workflow) {
          await bridgeRef.current.updateWorkflowVisualization(workflowData);
        } else {
          await bridgeRef.current.createWorkflowVisualization(workflowData);
        }
      }
    } catch (error: any) {
      console.error('Error fetching workflow:', error);
      setError(error.message || 'Failed to load workflow');
    }
  }, [workflowId, isInitialized, workflow]);

  // Initial workflow fetch
  useEffect(() => {
    fetchWorkflow();
  }, [fetchWorkflow]);

  // Real-time updates
  useEffect(() => {
    if (!workflowId) return;

    const handleSocketMessage = (eventData: any) => {
      if (eventData.workflow_id !== workflowId) return;

      switch (eventData.event_type) {
        case 'TASK_STATUS_UPDATE':
          handleTaskStatusUpdate(eventData);
          break;
        case 'WORKFLOW_COMPLETED':
          fetchWorkflow();
          break;
        case 'WORKFLOW_UPDATED':
          fetchWorkflow();
          break;
        case 'COLLABORATION_USER_JOINED':
          handleCollaborationUserJoined(eventData);
          break;
        case 'COLLABORATION_USER_LEFT':
          handleCollaborationUserLeft(eventData);
          break;
      }
    };

    connectToDashboardSocket(handleSocketMessage);

    return () => {
      disconnectFromDashboardSocket();
    };
  }, [workflowId, fetchWorkflow]);

  // Event handlers
  const handleTaskStatusUpdate = useCallback(async (eventData: any) => {
    if (!bridgeRef.current || !isInitialized) return;

    const status = {
      status: getStatusCode(eventData.status),
      progress: eventData.progress || 0,
      message: eventData.message || '',
      errorDetails: eventData.error_details || '',
      metadata: eventData.metadata || {}
    };

    await bridgeRef.current.updateNodeStatus(eventData.task_id, status);
  }, [isInitialized]);

  const handleCollaborationUserJoined = useCallback((eventData: any) => {
    if (!bridgeRef.current || !collaborationState.enabled) return;

    const user: UserPresence = {
      userId: eventData.user_id,
      displayName: eventData.display_name,
      avatarColor: eventData.avatar_color,
      position: { x: 0, y: 0, z: 0 },
      orientation: { x: 0, y: 0, z: 0 },
      isEditing: false,
      voiceActive: false,
      lastActivity: Date.now(),
      metadata: eventData.metadata || {}
    };

    bridgeRef.current.addCollaborativeUser(user);
  }, [collaborationState.enabled]);

  const handleCollaborationUserLeft = useCallback((eventData: any) => {
    if (!bridgeRef.current || !collaborationState.enabled) return;

    bridgeRef.current.removeCollaborativeUser(eventData.user_id);
  }, [collaborationState.enabled]);

  const handleUserInteraction = useCallback((interaction: any) => {
    switch (interaction.interactionType) {
      case 'click':
        if (interaction.targetNodeId) {
          handleNodeClick(interaction.targetNodeId);
        }
        break;
      case 'double_click':
        if (interaction.targetNodeId) {
          handleNodeDoubleClick(interaction.targetNodeId);
        }
        break;
      case 'edge_click':
        if (interaction.targetEdgeId) {
          handleEdgeClick(interaction.targetEdgeId);
        }
        break;
    }
  }, []);

  const handleNodeClick = useCallback((nodeId: string) => {
    if (!workflow) return;

    const task = findTaskById(workflow.tasks, nodeId);
    if (task) {
      setSelectedTask(task);
      setIsPanelOpen(true);
      if (onNodeClick) onNodeClick(nodeId);
    }
  }, [workflow, onNodeClick]);

  const handleNodeDoubleClick = useCallback((nodeId: string) => {
    if (onNodeDoubleClick) onNodeDoubleClick(nodeId);
  }, [onNodeDoubleClick]);

  const handleEdgeClick = useCallback((edgeId: string) => {
    if (onEdgeClick) onEdgeClick(edgeId);
  }, [onEdgeClick]);

  const handleClosePanel = useCallback(() => {
    setIsPanelOpen(false);
    setSelectedTask(null);
  }, []);

  // Visualization controls
  const handleModeChange = useCallback((mode: VisualizationMode) => {
    if (!bridgeRef.current) return;
    
    bridgeRef.current.setVisualizationMode(mode);
    setVisualizationState(prev => ({ ...prev, mode }));
  }, []);

  const handleQualityChange = useCallback((quality: QualityLevel) => {
    if (!bridgeRef.current) return;
    
    bridgeRef.current.setQualityLevel(quality);
    setVisualizationState(prev => ({ ...prev, quality }));
  }, []);

  const handleFullscreenToggle = useCallback(() => {
    if (!containerRef.current) return;

    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen();
      setVisualizationState(prev => ({ ...prev, isFullscreen: true }));
    } else {
      document.exitFullscreen();
      setVisualizationState(prev => ({ ...prev, isFullscreen: false }));
    }
  }, []);

  // Collaboration controls
  const handleCollaborationToggle = useCallback(async () => {
    if (!bridgeRef.current) return;

    if (!collaborationState.enabled) {
      const newSessionId = sessionId || `session_${Date.now()}`;
      await bridgeRef.current.enableCollaboration(newSessionId);
      setCollaborationState(prev => ({ ...prev, enabled: true, sessionId: newSessionId }));
    } else {
      bridgeRef.current.disableCollaboration();
      setCollaborationState(prev => ({ ...prev, enabled: false, sessionId: '', users: [] }));
    }
  }, [collaborationState.enabled, sessionId]);

  // Layout optimization
  const handleLayoutOptimization = useCallback(async () => {
    if (!bridgeRef.current || !workflow) return;

    try {
      const params = {
        workflowData: JSON.stringify(workflow),
        canvasSize: { x: canvasRef.current?.width || 800, y: canvasRef.current?.height || 600, z: 100 },
        layoutAlgorithm: 'ai_optimized',
        userPreferences: {},
        performanceConstraints: performanceState.metrics,
        enableCollaboration: collaborationState.enabled,
        maxIterations: 1000,
        convergenceThreshold: 0.001
      };

      await bridgeRef.current.optimizeLayout(params);
      setNotificationMessage('Layout optimization completed');
      setShowNotification(true);
    } catch (error) {
      console.error('Layout optimization failed:', error);
      setError('Layout optimization failed');
    }
  }, [workflow, performanceState.metrics, collaborationState.enabled]);

  // Utility functions
  const findTaskById = (tasks: TaskBlock[], taskId: string): TaskBlock | null => {
    for (const task of tasks) {
      if (task.task_id === taskId) {
        return task;
      }
      if (task.type === 'conditional' && task.branches) {
        for (const branch of task.branches) {
          const found = findTaskById(branch.tasks, taskId);
          if (found) return found;
        }
      }
      if (task.type === 'loop' && task.tasks) {
        const found = findTaskById(task.tasks, taskId);
        if (found) return found;
      }
    }
    return null;
  };

  const getStatusCode = (status: string): number => {
    switch (status) {
      case 'pending': return 0;
      case 'running': return 1;
      case 'completed': return 2;
      case 'failed': return 3;
      case 'pending_approval': return 4;
      case 'cancelled': return 5;
      default: return 0;
    }
  };

  const getQualityLabel = (quality: QualityLevel): string => {
    switch (quality) {
      case QualityLevel.Low: return 'Low';
      case QualityLevel.Medium: return 'Medium';
      case QualityLevel.High: return 'High';
      case QualityLevel.Ultra: return 'Ultra';
      default: return 'Medium';
    }
  };

  const getModeLabel = (mode: VisualizationMode): string => {
    switch (mode) {
      case VisualizationMode.TwoD: return '2D';
      case VisualizationMode.ThreeD: return '3D';
      case VisualizationMode.VR: return 'VR';
      case VisualizationMode.AR: return 'AR';
      default: return '3D';
    }
  };

  // Error handling
  if (error) {
    return (
      <Paper elevation={3} sx={{ height: '70vh', width: '100%', p: 2 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button variant="contained" onClick={fetchWorkflow}>
          Retry
        </Button>
      </Paper>
    );
  }

  return (
    <VisualizerContainer ref={containerRef}>
      {/* Loading overlay */}
      {loading && (
        <LoadingOverlay>
          <Box textAlign="center">
            <CircularProgress size={60} color="primary" />
            <Typography variant="h6" sx={{ mt: 2 }}>
              Initializing 3D Visualization...
            </Typography>
          </Box>
        </LoadingOverlay>
      )}

      {/* Main canvas */}
      <CanvasContainer>
        <canvas ref={canvasRef} />
      </CanvasContainer>

      {/* Control Panel */}
      <ControlPanel>
        {/* Mode controls */}
        <Box sx={{ p: 1 }}>
          <ButtonGroup variant="outlined" size="small">
            <Tooltip title="2D View">
              <Button 
                onClick={() => handleModeChange(VisualizationMode.TwoD)}
                variant={visualizationState.mode === VisualizationMode.TwoD ? 'contained' : 'outlined'}
              >
                <TwoD />
              </Button>
            </Tooltip>
            <Tooltip title="3D View">
              <Button 
                onClick={() => handleModeChange(VisualizationMode.ThreeD)}
                variant={visualizationState.mode === VisualizationMode.ThreeD ? 'contained' : 'outlined'}
              >
                <ThreeD />
              </Button>
            </Tooltip>
            {enableVR && (
              <Tooltip title="VR View">
                <Button 
                  onClick={() => handleModeChange(VisualizationMode.VR)}
                  variant={visualizationState.mode === VisualizationMode.VR ? 'contained' : 'outlined'}
                >
                  <ViewInAr />
                </Button>
              </Tooltip>
            )}
            {enableAR && (
              <Tooltip title="AR View">
                <Button 
                  onClick={() => handleModeChange(VisualizationMode.AR)}
                  variant={visualizationState.mode === VisualizationMode.AR ? 'contained' : 'outlined'}
                >
                  <ViewModule />
                </Button>
              </Tooltip>
            )}
          </ButtonGroup>
        </Box>

        {/* Quality controls */}
        <Box sx={{ p: 1 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Quality</InputLabel>
            <Select
              value={visualizationState.quality}
              label="Quality"
              onChange={(e) => handleQualityChange(e.target.value as QualityLevel)}
            >
              <MenuItem value={QualityLevel.Low}>Low</MenuItem>
              <MenuItem value={QualityLevel.Medium}>Medium</MenuItem>
              <MenuItem value={QualityLevel.High}>High</MenuItem>
              <MenuItem value={QualityLevel.Ultra}>Ultra</MenuItem>
            </Select>
          </FormControl>
        </Box>

        {/* Action buttons */}
        <Box sx={{ p: 1 }}>
          <ButtonGroup orientation="vertical" size="small">
            <Tooltip title="Optimize Layout">
              <Button onClick={handleLayoutOptimization} startIcon={<AutoAwesome />}>
                Optimize
              </Button>
            </Tooltip>
            <Tooltip title="Refresh">
              <Button onClick={fetchWorkflow} startIcon={<Refresh />}>
                Refresh
              </Button>
            </Tooltip>
            <Tooltip title="Settings">
              <Button onClick={() => setSettingsOpen(true)} startIcon={<Settings />}>
                Settings
              </Button>
            </Tooltip>
          </ButtonGroup>
        </Box>

        {/* Fullscreen toggle */}
        <Box sx={{ p: 1 }}>
          <IconButton onClick={handleFullscreenToggle} size="small">
            {visualizationState.isFullscreen ? <FullscreenExit /> : <Fullscreen />}
          </IconButton>
        </Box>
      </ControlPanel>

      {/* Collaboration Panel */}
      {enableCollaboration && (
        <CollaborationPanel>
          <Box sx={{ p: 1 }}>
            <ButtonGroup orientation="vertical" size="small">
              <Tooltip title="Toggle Collaboration">
                <Button
                  onClick={handleCollaborationToggle}
                  variant={collaborationState.enabled ? 'contained' : 'outlined'}
                  startIcon={<Share />}
                >
                  Collaborate
                </Button>
              </Tooltip>
              <Tooltip title="Voice Chat">
                <Button
                  disabled={!collaborationState.enabled}
                  variant={collaborationState.voiceEnabled ? 'contained' : 'outlined'}
                  startIcon={collaborationState.voiceEnabled ? <Mic /> : <MicOff />}
                >
                  Voice
                </Button>
              </Tooltip>
              <Tooltip title="Show Users">
                <Button
                  disabled={!collaborationState.enabled}
                  onClick={() => setCollaborationOpen(true)}
                  startIcon={
                    <Badge badgeContent={collaborationState.users.length} color="primary">
                      <Group />
                    </Badge>
                  }
                >
                  Users
                </Button>
              </Tooltip>
            </ButtonGroup>
          </Box>
        </CollaborationPanel>
      )}

      {/* Performance Panel */}
      {performanceState.showMetrics && (
        <PerformancePanel>
          <Typography variant="subtitle2" gutterBottom>
            Performance Metrics
          </Typography>
          <Box sx={{ fontSize: '0.75rem' }}>
            <Box display="flex" justifyContent="space-between">
              <span>FPS:</span>
              <span>{Math.round(1000 / performanceState.metrics.frameTime)}</span>
            </Box>
            <Box display="flex" justifyContent="space-between">
              <span>Render Time:</span>
              <span>{performanceState.metrics.renderTime.toFixed(2)}ms</span>
            </Box>
            <Box display="flex" justifyContent="space-between">
              <span>Draw Calls:</span>
              <span>{performanceState.metrics.drawCalls}</span>
            </Box>
            <Box display="flex" justifyContent="space-between">
              <span>Visible Nodes:</span>
              <span>{performanceState.metrics.visibleNodes}</span>
            </Box>
            <Box display="flex" justifyContent="space-between">
              <span>Memory:</span>
              <span>{(performanceState.metrics.memoryUsage / 1024 / 1024).toFixed(1)}MB</span>
            </Box>
          </Box>
        </PerformancePanel>
      )}

      {/* Status Bar */}
      <StatusBar>
        <Typography variant="caption" sx={{ flexGrow: 1 }}>
          {workflow ? `${workflow.name} - ${getModeLabel(visualizationState.mode)} - ${getQualityLabel(visualizationState.quality)}` : 'Loading...'}
        </Typography>
        <Chip 
          label={`${performanceState.metrics.activeNodes} nodes`} 
          size="small" 
          variant="outlined" 
          sx={{ mr: 1 }}
        />
        <Chip 
          label={`${Math.round(1000 / performanceState.metrics.frameTime)} FPS`} 
          size="small" 
          variant="outlined" 
          color={performanceState.metrics.frameTime > 33 ? 'warning' : 'success'}
        />
      </StatusBar>

      {/* Task Detail Panel */}
      <TaskDetailPanel
        task={selectedTask}
        open={isPanelOpen}
        onClose={handleClosePanel}
      />

      {/* Settings Drawer */}
      <Drawer
        anchor="right"
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        PaperProps={{ sx: { width: 320 } }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Visualization Settings
          </Typography>
          <Divider sx={{ mb: 2 }} />
          
          <FormControlLabel
            control={
              <Switch
                checked={visualizationState.showGrid}
                onChange={(e) => setVisualizationState(prev => ({ ...prev, showGrid: e.target.checked }))}
              />
            }
            label="Show Grid"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={visualizationState.showLabels}
                onChange={(e) => setVisualizationState(prev => ({ ...prev, showLabels: e.target.checked }))}
              />
            }
            label="Show Labels"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={visualizationState.showAnimations}
                onChange={(e) => setVisualizationState(prev => ({ ...prev, showAnimations: e.target.checked }))}
              />
            }
            label="Show Animations"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={visualizationState.showParticles}
                onChange={(e) => setVisualizationState(prev => ({ ...prev, showParticles: e.target.checked }))}
              />
            }
            label="Show Particles"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={visualizationState.enableSpatialAudio}
                onChange={(e) => setVisualizationState(prev => ({ ...prev, enableSpatialAudio: e.target.checked }))}
              />
            }
            label="Spatial Audio"
          />
          
          <Typography variant="subtitle2" sx={{ mt: 2 }}>
            Performance
          </Typography>
          
          <FormControlLabel
            control={
              <Switch
                checked={performanceState.showMetrics}
                onChange={(e) => setPerformanceState(prev => ({ ...prev, showMetrics: e.target.checked }))}
              />
            }
            label="Show Performance Metrics"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={performanceState.autoOptimize}
                onChange={(e) => setPerformanceState(prev => ({ ...prev, autoOptimize: e.target.checked }))}
              />
            }
            label="Auto-optimize Quality"
          />
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2">Target FPS: {performanceState.targetFPS}</Typography>
            <Slider
              value={performanceState.targetFPS}
              min={30}
              max={120}
              step={10}
              marks
              valueLabelDisplay="auto"
              onChange={(e, value) => setPerformanceState(prev => ({ ...prev, targetFPS: value as number }))}
            />
          </Box>
        </Box>
      </Drawer>

      {/* Collaboration Users Drawer */}
      <Drawer
        anchor="right"
        open={collaborationOpen}
        onClose={() => setCollaborationOpen(false)}
        PaperProps={{ sx: { width: 280 } }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Collaboration Users
          </Typography>
          <Divider sx={{ mb: 2 }} />
          
          {collaborationState.users.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No users connected
            </Typography>
          ) : (
            <List>
              {collaborationState.users.map((user) => (
                <ListItem key={user.userId}>
                  <AccountCircle sx={{ mr: 1, color: user.avatarColor }} />
                  <ListItemText
                    primary={user.displayName}
                    secondary={user.isEditing ? 'Editing' : 'Viewing'}
                  />
                  <ListItemSecondaryAction>
                    {user.voiceActive && <Mic fontSize="small" />}
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          )}
        </Box>
      </Drawer>

      {/* Notification Snackbar */}
      <Snackbar
        open={showNotification}
        autoHideDuration={3000}
        onClose={() => setShowNotification(false)}
        message={notificationMessage}
      />
    </VisualizerContainer>
  );
};

export default O3DEWorkflowVisualizer; 