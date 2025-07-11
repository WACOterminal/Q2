import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Switch,
  FormControlLabel,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Card,
  CardContent,
  CardActions,
  Chip,
  Alert,
  LinearProgress,
  Tabs,
  Tab,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  VolumeUp,
  VolumeOff,
  Accessibility,
  Speed,
  Visibility,
  VisibilityOff,
  Mic,
  MicOff,
  Gamepad2,
  Person,
  Settings,
  Analytics,
  Security,
  Memory,
  Cpu,
  GraphicEq,
  VrPano,
  ThreeD,
  TwoDimensional,
  ColorLens,
  Contrast,
  FontDownload,
  Keyboard,
  Mouse,
  VoiceChat,
  Vibration,
  RemoveRedEye,
  Hearing,
  TouchApp,
  Psychology,
  BarChart
} from '@mui/icons-material';
import { O3DEWorkflowBridge } from './O3DEWorkflowBridge';
import { O3DEWorkflowVisualizer } from './O3DEWorkflowVisualizer';

interface O3DEIntegrationDemoProps {
  workflowId: string;
  onFeatureToggle?: (feature: string, enabled: boolean) => void;
  onSettingsChange?: (settings: any) => void;
}

interface PerformanceMetrics {
  frameTime: number;
  renderTime: number;
  updateTime: number;
  memoryUsage: number;
  cpuUsage: number;
  gpuUsage: number;
  drawCalls: number;
  vertices: number;
  triangles: number;
  networkLatency: number;
  qualityLevel: string;
  visibleNodes: number;
  activeNodes: number;
}

interface AccessibilitySettings {
  screenReader: boolean;
  keyboardNavigation: boolean;
  voiceCommands: boolean;
  hapticFeedback: boolean;
  highContrast: boolean;
  colorBlindSupport: boolean;
  motionReduction: boolean;
  fontScaling: number;
  uiScaling: number;
  audioDescriptions: boolean;
  spatialAudio: boolean;
  simplifiedInterface: boolean;
  cognitiveAids: boolean;
  gestureAlternatives: boolean;
  eyeTracking: boolean;
}

interface CollaborationSettings {
  enabled: boolean;
  sessionId: string;
  spatialAudio: boolean;
  voiceChat: boolean;
  screenSharing: boolean;
  fileTransfer: boolean;
  multiUserEditing: boolean;
  presenceIndicators: boolean;
  chatIntegration: boolean;
  permissionManagement: boolean;
}

interface AIOptimizationSettings {
  layoutOptimization: boolean;
  performanceOptimization: boolean;
  userPersonalization: boolean;
  contextualAssistance: boolean;
  smartSuggestions: boolean;
  automaticRecommendations: boolean;
  learningMode: boolean;
  adaptiveInterface: boolean;
}

export const O3DEIntegrationDemo: React.FC<O3DEIntegrationDemoProps> = ({
  workflowId,
  onFeatureToggle,
  onSettingsChange
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [o3deBridge, setO3deBridge] = useState<O3DEWorkflowBridge | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Core settings
  const [visualizationMode, setVisualizationMode] = useState('3d');
  const [qualityLevel, setQualityLevel] = useState('medium');
  const [renderingEngine, setRenderingEngine] = useState('o3de');

  // Performance metrics
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    frameTime: 16.67,
    renderTime: 8.5,
    updateTime: 3.2,
    memoryUsage: 256,
    cpuUsage: 35,
    gpuUsage: 28,
    drawCalls: 150,
    vertices: 25000,
    triangles: 12500,
    networkLatency: 45,
    qualityLevel: 'medium',
    visibleNodes: 25,
    activeNodes: 50
  });

  // Accessibility settings
  const [accessibilitySettings, setAccessibilitySettings] = useState<AccessibilitySettings>({
    screenReader: false,
    keyboardNavigation: true,
    voiceCommands: false,
    hapticFeedback: false,
    highContrast: false,
    colorBlindSupport: false,
    motionReduction: false,
    fontScaling: 1.0,
    uiScaling: 1.0,
    audioDescriptions: false,
    spatialAudio: false,
    simplifiedInterface: false,
    cognitiveAids: false,
    gestureAlternatives: false,
    eyeTracking: false
  });

  // Collaboration settings
  const [collaborationSettings, setCollaborationSettings] = useState<CollaborationSettings>({
    enabled: false,
    sessionId: '',
    spatialAudio: false,
    voiceChat: false,
    screenSharing: false,
    fileTransfer: false,
    multiUserEditing: false,
    presenceIndicators: true,
    chatIntegration: false,
    permissionManagement: true
  });

  // AI optimization settings
  const [aiOptimizationSettings, setAIOptimizationSettings] = useState<AIOptimizationSettings>({
    layoutOptimization: true,
    performanceOptimization: true,
    userPersonalization: false,
    contextualAssistance: false,
    smartSuggestions: true,
    automaticRecommendations: false,
    learningMode: false,
    adaptiveInterface: false
  });

  // Initialize O3DE integration
  useEffect(() => {
    const initializeO3DE = async () => {
      try {
        setIsLoading(true);
        
        // Initialize the O3DE bridge
        const bridge = new O3DEWorkflowBridge();
        const initialized = await bridge.initialize();
        
        if (!initialized) {
          throw new Error('Failed to initialize O3DE bridge');
        }

        // Set up event handlers
        bridge.onPerformanceMetrics = (metrics: PerformanceMetrics) => {
          setPerformanceMetrics(metrics);
        };

        bridge.onAccessibilityEvent = (event: any) => {
          console.log('Accessibility event:', event);
        };

        bridge.onCollaborationEvent = (event: any) => {
          console.log('Collaboration event:', event);
        };

        bridge.onAIOptimizationEvent = (event: any) => {
          console.log('AI optimization event:', event);
        };

        setO3deBridge(bridge);
        setIsInitialized(true);
        setError(null);
      } catch (err) {
        console.error('O3DE initialization error:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setIsLoading(false);
      }
    };

    initializeO3DE();

    return () => {
      if (o3deBridge) {
        o3deBridge.shutdown();
      }
    };
  }, []);

  // Update O3DE settings when they change
  useEffect(() => {
    if (o3deBridge && isInitialized) {
      o3deBridge.setVisualizationMode(visualizationMode);
      o3deBridge.setQualityLevel(qualityLevel);
      
      // Apply accessibility settings
      Object.entries(accessibilitySettings).forEach(([key, value]) => {
        o3deBridge.setAccessibilityFeature(key, value);
      });

      // Apply collaboration settings
      if (collaborationSettings.enabled) {
        o3deBridge.enableCollaboration(collaborationSettings.sessionId);
      }

      // Apply AI optimization settings
      Object.entries(aiOptimizationSettings).forEach(([key, value]) => {
        o3deBridge.setAIOptimizationFeature(key, value);
      });
    }
  }, [
    o3deBridge,
    isInitialized,
    visualizationMode,
    qualityLevel,
    accessibilitySettings,
    collaborationSettings,
    aiOptimizationSettings
  ]);

  const handleAccessibilityChange = (feature: keyof AccessibilitySettings, value: any) => {
    setAccessibilitySettings(prev => ({
      ...prev,
      [feature]: value
    }));
    onFeatureToggle?.(feature, value);
  };

  const handleCollaborationChange = (feature: keyof CollaborationSettings, value: any) => {
    setCollaborationSettings(prev => ({
      ...prev,
      [feature]: value
    }));
    onFeatureToggle?.(feature, value);
  };

  const handleAIOptimizationChange = (feature: keyof AIOptimizationSettings, value: any) => {
    setAIOptimizationSettings(prev => ({
      ...prev,
      [feature]: value
    }));
    onFeatureToggle?.(feature, value);
  };

  const getPerformanceColor = (value: number, threshold: number) => {
    if (value > threshold * 1.5) return 'error';
    if (value > threshold) return 'warning';
    return 'success';
  };

  const renderVisualizationTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={8}>
        <Paper sx={{ p: 2, height: '600px' }}>
          <Typography variant="h6" gutterBottom>
            3D Workflow Visualization
          </Typography>
          {isInitialized ? (
            <O3DEWorkflowVisualizer
              workflowId={workflowId}
              bridge={o3deBridge}
              visualizationMode={visualizationMode}
              qualityLevel={qualityLevel}
              accessibilitySettings={accessibilitySettings}
              collaborationSettings={collaborationSettings}
              aiOptimizationSettings={aiOptimizationSettings}
            />
          ) : (
            <Box display="flex" justifyContent="center" alignItems="center" height="100%">
              <Typography variant="body1" color="text.secondary">
                {isLoading ? 'Loading O3DE...' : 'O3DE not initialized'}
              </Typography>
            </Box>
          )}
        </Paper>
      </Grid>

      <Grid item xs={12} md={4}>
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="h6" gutterBottom>
            Visualization Settings
          </Typography>
          
          <FormControl fullWidth margin="normal">
            <InputLabel>Visualization Mode</InputLabel>
            <Select
              value={visualizationMode}
              onChange={(e) => setVisualizationMode(e.target.value)}
            >
              <MenuItem value="2d">2D View</MenuItem>
              <MenuItem value="3d">3D View</MenuItem>
              <MenuItem value="vr">VR Mode</MenuItem>
              <MenuItem value="ar">AR Mode</MenuItem>
            </Select>
          </FormControl>

          <FormControl fullWidth margin="normal">
            <InputLabel>Quality Level</InputLabel>
            <Select
              value={qualityLevel}
              onChange={(e) => setQualityLevel(e.target.value)}
            >
              <MenuItem value="low">Low</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="high">High</MenuItem>
              <MenuItem value="ultra">Ultra</MenuItem>
            </Select>
          </FormControl>

          <FormControl fullWidth margin="normal">
            <InputLabel>Rendering Engine</InputLabel>
            <Select
              value={renderingEngine}
              onChange={(e) => setRenderingEngine(e.target.value)}
            >
              <MenuItem value="o3de">O3DE Engine</MenuItem>
              <MenuItem value="webgl">WebGL</MenuItem>
              <MenuItem value="canvas">Canvas 2D</MenuItem>
            </Select>
          </FormControl>
        </Paper>

        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Quick Actions
          </Typography>
          
          <Box display="flex" flexDirection="column" gap={1}>
            <Button
              variant="outlined"
              startIcon={<Analytics />}
              onClick={() => o3deBridge?.optimizeLayout()}
            >
              Optimize Layout
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<Speed />}
              onClick={() => o3deBridge?.optimizePerformance()}
            >
              Optimize Performance
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<Accessibility />}
              onClick={() => o3deBridge?.validateAccessibility()}
            >
              Validate Accessibility
            </Button>
            
            <Button
              variant="outlined"
              startIcon={<Security />}
              onClick={() => o3deBridge?.runDiagnostics()}
            >
              Run Diagnostics
            </Button>
          </Box>
        </Paper>
      </Grid>
    </Grid>
  );

  const renderPerformanceTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Real-time Performance Metrics
          </Typography>
          
          <Box mb={2}>
            <Typography variant="body2" color="text.secondary">
              Frame Time: {performanceMetrics.frameTime.toFixed(2)}ms
            </Typography>
            <LinearProgress
              variant="determinate"
              value={Math.min((performanceMetrics.frameTime / 33.33) * 100, 100)}
              color={getPerformanceColor(performanceMetrics.frameTime, 16.67)}
            />
          </Box>

          <Box mb={2}>
            <Typography variant="body2" color="text.secondary">
              CPU Usage: {performanceMetrics.cpuUsage.toFixed(1)}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={performanceMetrics.cpuUsage}
              color={getPerformanceColor(performanceMetrics.cpuUsage, 70)}
            />
          </Box>

          <Box mb={2}>
            <Typography variant="body2" color="text.secondary">
              GPU Usage: {performanceMetrics.gpuUsage.toFixed(1)}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={performanceMetrics.gpuUsage}
              color={getPerformanceColor(performanceMetrics.gpuUsage, 80)}
            />
          </Box>

          <Box mb={2}>
            <Typography variant="body2" color="text.secondary">
              Memory Usage: {performanceMetrics.memoryUsage.toFixed(0)}MB
            </Typography>
            <LinearProgress
              variant="determinate"
              value={Math.min((performanceMetrics.memoryUsage / 1024) * 100, 100)}
              color={getPerformanceColor(performanceMetrics.memoryUsage, 512)}
            />
          </Box>
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Rendering Statistics
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h4" color="primary">
                    {performanceMetrics.drawCalls}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Draw Calls
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h4" color="primary">
                    {(performanceMetrics.vertices / 1000).toFixed(1)}K
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Vertices
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h4" color="primary">
                    {(performanceMetrics.triangles / 1000).toFixed(1)}K
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Triangles
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h4" color="primary">
                    {performanceMetrics.visibleNodes}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Visible Nodes
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Performance Optimization
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={true}
                    onChange={(e) => o3deBridge?.setPerformanceFeature('lod', e.target.checked)}
                  />
                }
                label="Level of Detail (LOD)"
              />
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={true}
                    onChange={(e) => o3deBridge?.setPerformanceFeature('culling', e.target.checked)}
                  />
                }
                label="Frustum Culling"
              />
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={true}
                    onChange={(e) => o3deBridge?.setPerformanceFeature('batching', e.target.checked)}
                  />
                }
                label="Batch Rendering"
              />
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControlLabel
                control={
                  <Switch
                    checked={true}
                    onChange={(e) => o3deBridge?.setPerformanceFeature('adaptive', e.target.checked)}
                  />
                }
                label="Adaptive Quality"
              />
            </Grid>
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );

  const renderAccessibilityTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            <Accessibility sx={{ mr: 1 }} />
            Visual Accessibility
          </Typography>
          
          <FormControlLabel
            control={
              <Switch
                checked={accessibilitySettings.screenReader}
                onChange={(e) => handleAccessibilityChange('screenReader', e.target.checked)}
              />
            }
            label="Screen Reader Support"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={accessibilitySettings.keyboardNavigation}
                onChange={(e) => handleAccessibilityChange('keyboardNavigation', e.target.checked)}
              />
            }
            label="Keyboard Navigation"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={accessibilitySettings.highContrast}
                onChange={(e) => handleAccessibilityChange('highContrast', e.target.checked)}
              />
            }
            label="High Contrast Mode"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={accessibilitySettings.colorBlindSupport}
                onChange={(e) => handleAccessibilityChange('colorBlindSupport', e.target.checked)}
              />
            }
            label="Color Blind Support"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={accessibilitySettings.motionReduction}
                onChange={(e) => handleAccessibilityChange('motionReduction', e.target.checked)}
              />
            }
            label="Reduced Motion"
          />
          
          <Box mt={2}>
            <Typography variant="body2" gutterBottom>
              Font Scaling: {accessibilitySettings.fontScaling.toFixed(1)}x
            </Typography>
            <Slider
              value={accessibilitySettings.fontScaling}
              onChange={(_, value) => handleAccessibilityChange('fontScaling', value)}
              min={0.5}
              max={2.0}
              step={0.1}
              marks
              valueLabelDisplay="auto"
            />
          </Box>
          
          <Box mt={2}>
            <Typography variant="body2" gutterBottom>
              UI Scaling: {accessibilitySettings.uiScaling.toFixed(1)}x
            </Typography>
            <Slider
              value={accessibilitySettings.uiScaling}
              onChange={(_, value) => handleAccessibilityChange('uiScaling', value)}
              min={0.5}
              max={2.0}
              step={0.1}
              marks
              valueLabelDisplay="auto"
            />
          </Box>
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            <VoiceChat sx={{ mr: 1 }} />
            Audio & Motor Accessibility
          </Typography>
          
          <FormControlLabel
            control={
              <Switch
                checked={accessibilitySettings.voiceCommands}
                onChange={(e) => handleAccessibilityChange('voiceCommands', e.target.checked)}
              />
            }
            label="Voice Commands"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={accessibilitySettings.audioDescriptions}
                onChange={(e) => handleAccessibilityChange('audioDescriptions', e.target.checked)}
              />
            }
            label="Audio Descriptions"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={accessibilitySettings.spatialAudio}
                onChange={(e) => handleAccessibilityChange('spatialAudio', e.target.checked)}
              />
            }
            label="Spatial Audio"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={accessibilitySettings.hapticFeedback}
                onChange={(e) => handleAccessibilityChange('hapticFeedback', e.target.checked)}
              />
            }
            label="Haptic Feedback"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={accessibilitySettings.gestureAlternatives}
                onChange={(e) => handleAccessibilityChange('gestureAlternatives', e.target.checked)}
              />
            }
            label="Gesture Alternatives"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={accessibilitySettings.eyeTracking}
                onChange={(e) => handleAccessibilityChange('eyeTracking', e.target.checked)}
              />
            }
            label="Eye Tracking"
          />
        </Paper>
      </Grid>

      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            <Psychology sx={{ mr: 1 }} />
            Cognitive Accessibility
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={accessibilitySettings.simplifiedInterface}
                    onChange={(e) => handleAccessibilityChange('simplifiedInterface', e.target.checked)}
                  />
                }
                label="Simplified Interface"
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={accessibilitySettings.cognitiveAids}
                    onChange={(e) => handleAccessibilityChange('cognitiveAids', e.target.checked)}
                  />
                }
                label="Cognitive Aids"
              />
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Button
                variant="outlined"
                startIcon={<Analytics />}
                onClick={() => o3deBridge?.validateAccessibility()}
                fullWidth
              >
                Validate Compliance
              </Button>
            </Grid>
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );

  const renderCollaborationTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            <Person sx={{ mr: 1 }} />
            Collaboration Features
          </Typography>
          
          <FormControlLabel
            control={
              <Switch
                checked={collaborationSettings.enabled}
                onChange={(e) => handleCollaborationChange('enabled', e.target.checked)}
              />
            }
            label="Enable Collaboration"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={collaborationSettings.spatialAudio}
                onChange={(e) => handleCollaborationChange('spatialAudio', e.target.checked)}
              />
            }
            label="Spatial Audio"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={collaborationSettings.voiceChat}
                onChange={(e) => handleCollaborationChange('voiceChat', e.target.checked)}
              />
            }
            label="Voice Chat"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={collaborationSettings.screenSharing}
                onChange={(e) => handleCollaborationChange('screenSharing', e.target.checked)}
              />
            }
            label="Screen Sharing"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={collaborationSettings.multiUserEditing}
                onChange={(e) => handleCollaborationChange('multiUserEditing', e.target.checked)}
              />
            }
            label="Multi-User Editing"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={collaborationSettings.presenceIndicators}
                onChange={(e) => handleCollaborationChange('presenceIndicators', e.target.checked)}
              />
            }
            label="Presence Indicators"
          />
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Active Collaboration Session
          </Typography>
          
          {collaborationSettings.enabled ? (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Session ID: {collaborationSettings.sessionId || 'Generating...'}
              </Typography>
              
              <Box display="flex" gap={1} mb={2}>
                <Chip
                  label="Host"
                  color="primary"
                  icon={<Person />}
                />
                <Chip
                  label="2 Participants"
                  color="secondary"
                  icon={<Person />}
                />
              </Box>
              
              <Alert severity="info" sx={{ mb: 2 }}>
                Real-time collaboration is active. All changes are synchronized across participants.
              </Alert>
              
              <Button
                variant="contained"
                color="error"
                onClick={() => handleCollaborationChange('enabled', false)}
                fullWidth
              >
                End Session
              </Button>
            </Box>
          ) : (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Start a collaboration session to work with others in real-time.
              </Typography>
              
              <Button
                variant="contained"
                color="primary"
                onClick={() => {
                  const sessionId = 'session_' + Date.now();
                  setCollaborationSettings(prev => ({
                    ...prev,
                    sessionId,
                    enabled: true
                  }));
                }}
                fullWidth
              >
                Start Collaboration
              </Button>
            </Box>
          )}
        </Paper>
      </Grid>
    </Grid>
  );

  const renderAIOptimizationTab = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            <Psychology sx={{ mr: 1 }} />
            AI-Powered Features
          </Typography>
          
          <FormControlLabel
            control={
              <Switch
                checked={aiOptimizationSettings.layoutOptimization}
                onChange={(e) => handleAIOptimizationChange('layoutOptimization', e.target.checked)}
              />
            }
            label="Layout Optimization"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={aiOptimizationSettings.performanceOptimization}
                onChange={(e) => handleAIOptimizationChange('performanceOptimization', e.target.checked)}
              />
            }
            label="Performance Optimization"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={aiOptimizationSettings.userPersonalization}
                onChange={(e) => handleAIOptimizationChange('userPersonalization', e.target.checked)}
              />
            }
            label="User Personalization"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={aiOptimizationSettings.contextualAssistance}
                onChange={(e) => handleAIOptimizationChange('contextualAssistance', e.target.checked)}
              />
            }
            label="Contextual Assistance"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={aiOptimizationSettings.smartSuggestions}
                onChange={(e) => handleAIOptimizationChange('smartSuggestions', e.target.checked)}
              />
            }
            label="Smart Suggestions"
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={aiOptimizationSettings.adaptiveInterface}
                onChange={(e) => handleAIOptimizationChange('adaptiveInterface', e.target.checked)}
              />
            }
            label="Adaptive Interface"
          />
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            AI Optimization Results
          </Typography>
          
          <Box mb={2}>
            <Typography variant="body2" color="text.secondary">
              Layout Efficiency: 87%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={87}
              color="success"
            />
          </Box>
          
          <Box mb={2}>
            <Typography variant="body2" color="text.secondary">
              Performance Score: 92%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={92}
              color="success"
            />
          </Box>
          
          <Box mb={2}>
            <Typography variant="body2" color="text.secondary">
              User Satisfaction: 95%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={95}
              color="success"
            />
          </Box>
          
          <Alert severity="success" sx={{ mb: 2 }}>
            AI optimization has improved overall system performance by 23%.
          </Alert>
          
          <Button
            variant="contained"
            color="primary"
            startIcon={<Analytics />}
            onClick={() => o3deBridge?.runAIOptimization()}
            fullWidth
          >
            Run Full AI Optimization
          </Button>
        </Paper>
      </Grid>
    </Grid>
  );

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          O3DE Integration Error
        </Typography>
        <Typography variant="body2">
          {error}
        </Typography>
      </Alert>
    );
  }

  return (
    <Box sx={{ width: '100%', typography: 'body1' }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          O3DE Workflow Visualization Integration
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Advanced 3D workflow visualization with immersive collaboration, AI optimization, and comprehensive accessibility features.
        </Typography>
      </Box>

      {isLoading && (
        <Box sx={{ mb: 2 }}>
          <LinearProgress />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Initializing O3DE engine and loading components...
          </Typography>
        </Box>
      )}

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
          <Tab icon={<ThreeD />} label="Visualization" />
          <Tab icon={<BarChart />} label="Performance" />
          <Tab icon={<Accessibility />} label="Accessibility" />
          <Tab icon={<Person />} label="Collaboration" />
          <Tab icon={<Psychology />} label="AI Optimization" />
        </Tabs>
      </Box>

      <Box sx={{ minHeight: '600px' }}>
        {activeTab === 0 && renderVisualizationTab()}
        {activeTab === 1 && renderPerformanceTab()}
        {activeTab === 2 && renderAccessibilityTab()}
        {activeTab === 3 && renderCollaborationTab()}
        {activeTab === 4 && renderAIOptimizationTab()}
      </Box>

      <Box sx={{ mt: 3, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
        <Typography variant="h6" gutterBottom>
          System Status
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <Chip
              label={`O3DE: ${isInitialized ? 'Running' : 'Initializing'}`}
              color={isInitialized ? 'success' : 'default'}
              icon={<Memory />}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <Chip
              label={`Quality: ${qualityLevel}`}
              color="info"
              icon={<GraphicEq />}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <Chip
              label={`Mode: ${visualizationMode.toUpperCase()}`}
              color="secondary"
              icon={visualizationMode === '3d' ? <ThreeD /> : <TwoDimensional />}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <Chip
              label={`FPS: ${(1000 / performanceMetrics.frameTime).toFixed(0)}`}
              color={performanceMetrics.frameTime < 20 ? 'success' : 'warning'}
              icon={<Speed />}
            />
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
};

export default O3DEIntegrationDemo; 