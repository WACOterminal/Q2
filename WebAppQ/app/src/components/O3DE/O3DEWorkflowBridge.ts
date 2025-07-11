import { EventEmitter } from 'events';
import { Workflow, TaskBlock } from '../../services/types';
import { produce } from 'immer';

// --- Data Structures ---
export interface Vector3 { x: number; y: number; z: number; }
export interface Node {
    id: string;
    label: string;
    type: string;
    parent?: string;
    state: string;
    position: Vector3;
    metadata: Record<string, any>;
}
export interface Link {
    id: string;
    source: string;
    target:string;
    state: string;
}

export interface WorldState {
    nodes: Record<string, Node>;
    links: Record<string, Link>;
}

type WorldUpdateCallback = (state: WorldState) => void;

// --- O3DE Workflow Bridge ---
class O3DEWorkflowBridge {
    private ws: WebSocket | null = null;
    private worldState: WorldState = { nodes: {}, links: {} };
    private onUpdateCallback: WorldUpdateCallback | null = null;

    public connect(url: string) {
        if (this.ws) {
            console.log("WebSocket already connected.");
            return;
        }

        console.log(`Connecting to WebSocket at ${url}...`);
        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
            console.log("WebSocket connection established.");
        };

        this.ws.onmessage = (event) => {
            this.handleMessage(event.data);
        };

        this.ws.onerror = (error) => {
            console.error("WebSocket error:", error);
        };

        this.ws.onclose = () => {
            console.log("WebSocket connection closed.");
            this.ws = null;
            // Optional: Implement reconnection logic here
        };
    }

    public disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }

    public onUpdate(callback: WorldUpdateCallback) {
        this.onUpdateCallback = callback;
    }

    private handleMessage(data: string) {
        try {
            const message = JSON.parse(data);
            let stateChanged = false;

            switch (message.type) {
                case "SNAPSHOT":
                    this.worldState = { nodes: {}, links: {} }; // Reset state
                    message.payload.nodes.forEach((node: Node) => {
                        this.worldState.nodes[node.id] = node;
                    });
                    message.payload.links.forEach((link: Link) => {
                        this.worldState.links[link.id] = link;
                    });
                    stateChanged = true;
                    break;
                
                case "TICK":
                    this.worldState = produce(this.worldState, draftState => {
                        message.payload.forEach((event: any) => {
                            switch(event.event_type) {
                                case "NODE_CREATED":
                                    draftState.nodes[event.data.id] = event.data;
                                    break;
                                case "LINK_CREATED":
                                    draftState.links[event.data.id] = event.data;
                                    break;
                                case "NODE_STATE_CHANGED":
                                    if (draftState.nodes[event.data.id]) {
                                        draftState.nodes[event.data.id].state = event.data.state;
                                    }
                                    break;
                                case "LINK_PULSE":
                                    // This is a transient event, can be handled by the visualizer
                                    // Or we can add temporary links to the state
                                    break;
                            }
                        });
                    });
                    stateChanged = true;
                    break;
            }

            if (stateChanged && this.onUpdateCallback) {
                this.onUpdateCallback(this.worldState);
            }
        } catch (error) {
            console.error("Failed to parse WebSocket message:", error);
        }
    }
}

// Singleton instance of the bridge
export const workflowBridge = new O3DEWorkflowBridge();

// O3DE WebAssembly module interface
interface O3DEModule {
  // Core workflow methods
  CreateWorkflowVisualization: (data: string) => void;
  UpdateWorkflowVisualization: (data: string) => void;
  UpdateNodeStatus: (nodeId: string, status: string) => void;
  
  // Visualization control
  SetVisualizationMode: (mode: number) => void;
  SetQualityLevel: (level: number) => void;
  
  // Collaboration methods
  EnableCollaboration: (enabled: boolean, sessionId: string) => void;
  AddCollaborativeUser: (userPresence: string) => void;
  RemoveCollaborativeUser: (userId: string) => void;
  UpdateUserCursor: (userId: string, x: number, y: number, z: number) => void;
  
  // Interaction handling
  HandleUserInteraction: (interaction: string) => void;
  
  // Layout optimization
  OptimizeLayout: (params: string) => string;
  
  // Performance monitoring
  GetPerformanceMetrics: () => string;
  IsSystemInitialized: () => boolean;
  
  // WebAssembly memory management
  _malloc: (size: number) => number;
  _free: (ptr: number) => void;
  
  // Canvas management
  canvas: HTMLCanvasElement;
  
  // Module lifecycle
  onRuntimeInitialized?: () => void;
  onAbort?: (what: any) => void;
}

// Type definitions for O3DE integration
export enum VisualizationMode {
  TwoD = 0,
  ThreeD = 1,
  VR = 2,
  AR = 3
}

export enum QualityLevel {
  Low = 0,
  Medium = 1,
  High = 2,
  Ultra = 3
}

export interface UserPresence {
  userId: string;
  displayName: string;
  avatarColor: string;
  position: Vector3;
  orientation: Vector3;
  selectedNodeId?: string;
  isEditing: boolean;
  voiceActive: boolean;
  lastActivity: number;
  metadata: Record<string, string>;
}

export interface UserInteraction {
  interactionId: string;
  userId: string;
  interactionType: string;
  position: Vector3;
  direction: Vector3;
  targetNodeId: string;
  parameters: Record<string, string>;
  timestamp: number;
}

export interface NodeStatus {
  status: number;
  progress: number;
  message: string;
  errorDetails: string;
  metadata: Record<string, string>;
}

export interface PerformanceMetrics {
  frameTime: number;
  renderTime: number;
  updateTime: number;
  drawCalls: number;
  vertices: number;
  triangles: number;
  memoryUsage: number;
  gpuMemoryUsage: number;
  activeNodes: number;
  visibleNodes: number;
  currentQuality: QualityLevel;
  cpuUsage: number;
  gpuUsage: number;
  networkLatency: number;
  customMetrics: Record<string, number>;
}

export interface LayoutOptimizationParams {
  workflowData: string;
  canvasSize: Vector3;
  layoutAlgorithm: string;
  userPreferences: Record<string, string>;
  performanceConstraints: PerformanceMetrics;
  enableCollaboration: boolean;
  maxIterations: number;
  convergenceThreshold: number;
}

export interface OptimizedLayout {
  layoutId: string;
  algorithm: string;
  nodePositions: Record<string, Vector3>;
  edgePaths: Record<string, Vector3[]>;
  cameraPosition: Vector3;
  cameraTarget: Vector3;
  recommendedQuality: QualityLevel;
  optimizationScore: number;
  layoutMetrics: Record<string, number>;
}

// Performance monitoring class
class PerformanceMonitor {
  private metrics: PerformanceMetrics;
  private updateInterval: number;
  private bridge: O3DEWorkflowBridge;
  private intervalId?: NodeJS.Timeout;

  constructor(bridge: O3DEWorkflowBridge) {
    this.bridge = bridge;
    this.updateInterval = 1000; // 1 second
    this.metrics = {
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
      currentQuality: QualityLevel.Medium,
      cpuUsage: 0,
      gpuUsage: 0,
      networkLatency: 0,
      customMetrics: {}
    };
  }

  start(): void {
    this.intervalId = setInterval(() => {
      this.updateMetrics();
    }, this.updateInterval);
  }

  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
  }

  private updateMetrics(): void {
    if (this.bridge.isInitialized()) {
      try {
        const metricsJson = this.bridge.module?.GetPerformanceMetrics();
        if (metricsJson) {
          this.metrics = JSON.parse(metricsJson);
          this.bridge.emit('performanceMetrics', this.metrics);
        }
      } catch (error) {
        console.error('Failed to update performance metrics:', error);
      }
    }
  }

  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  setUpdateInterval(interval: number): void {
    this.updateInterval = interval;
    if (this.intervalId) {
      this.stop();
      this.start();
    }
  }
}

// Collaboration manager class
class CollaborationManager {
  private users: Map<string, UserPresence> = new Map();
  private sessionId: string = '';
  private userId: string = '';
  private enabled: boolean = false;
  private bridge: O3DEWorkflowBridge;
  private webrtcConnection?: RTCPeerConnection;
  private dataChannel?: RTCDataChannel;

  constructor(bridge: O3DEWorkflowBridge) {
    this.bridge = bridge;
  }

  async initialize(sessionId: string, userId: string): Promise<void> {
    this.sessionId = sessionId;
    this.userId = userId;
    this.enabled = true;

    // Initialize WebRTC for collaboration
    await this.setupWebRTC();

    // Enable collaboration in O3DE
    if (this.bridge.isInitialized()) {
      this.bridge.module?.EnableCollaboration(true, sessionId);
    }
  }

  private async setupWebRTC(): Promise<void> {
    try {
      this.webrtcConnection = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          // Add TURN servers for production
        ]
      });

      this.dataChannel = this.webrtcConnection.createDataChannel('collaboration', {
        ordered: true
      });

      this.dataChannel.addEventListener('message', (event) => {
        this.handleCollaborationMessage(event.data);
      });

      this.dataChannel.addEventListener('open', () => {
        console.log('Collaboration data channel opened');
      });

      this.dataChannel.addEventListener('error', (error) => {
        console.error('Collaboration data channel error:', error);
      });

    } catch (error) {
      console.error('Failed to setup WebRTC:', error);
    }
  }

  private handleCollaborationMessage(data: string): void {
    try {
      const message = JSON.parse(data);
      
      switch (message.type) {
        case 'user_join':
          this.handleUserJoin(message.data);
          break;
        case 'user_leave':
          this.handleUserLeave(message.data);
          break;
        case 'cursor_move':
          this.handleCursorMove(message.data);
          break;
        case 'node_interaction':
          this.handleNodeInteraction(message.data);
          break;
        case 'voice_data':
          this.handleVoiceData(message.data);
          break;
      }
    } catch (error) {
      console.error('Failed to handle collaboration message:', error);
    }
  }

  private handleUserJoin(userData: any): void {
    const user: UserPresence = {
      userId: userData.userId,
      displayName: userData.displayName,
      avatarColor: userData.avatarColor,
      position: userData.position,
      orientation: userData.orientation,
      isEditing: false,
      voiceActive: false,
      lastActivity: Date.now(),
      metadata: userData.metadata || {}
    };

    this.users.set(user.userId, user);
    
    if (this.bridge.isInitialized()) {
      this.bridge.module?.AddCollaborativeUser(JSON.stringify(user));
    }

    this.bridge.emit('userJoined', user);
  }

  private handleUserLeave(userData: any): void {
    const userId = userData.userId;
    this.users.delete(userId);

    if (this.bridge.isInitialized()) {
      this.bridge.module?.RemoveCollaborativeUser(userId);
    }

    this.bridge.emit('userLeft', userId);
  }

  private handleCursorMove(data: any): void {
    const userId = data.userId;
    const position = data.position;

    const user = this.users.get(userId);
    if (user) {
      user.position = position;
      user.lastActivity = Date.now();

      if (this.bridge.isInitialized()) {
        this.bridge.module?.UpdateUserCursor(userId, position.x, position.y, position.z);
      }

      this.bridge.emit('userCursorMoved', { userId, position });
    }
  }

  private handleNodeInteraction(data: any): void {
    this.bridge.emit('nodeInteraction', data);
  }

  private handleVoiceData(data: any): void {
    this.bridge.emit('voiceData', data);
  }

  broadcastInteraction(interaction: UserInteraction): void {
    if (this.dataChannel && this.dataChannel.readyState === 'open') {
      const message = {
        type: 'node_interaction',
        data: interaction
      };
      this.dataChannel.send(JSON.stringify(message));
    }
  }

  broadcastCursorMove(position: Vector3): void {
    if (this.dataChannel && this.dataChannel.readyState === 'open') {
      const message = {
        type: 'cursor_move',
        data: { userId: this.userId, position }
      };
      this.dataChannel.send(JSON.stringify(message));
    }
  }

  getUsers(): UserPresence[] {
    return Array.from(this.users.values());
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  disable(): void {
    this.enabled = false;
    this.users.clear();
    
    if (this.webrtcConnection) {
      this.webrtcConnection.close();
      this.webrtcConnection = undefined;
    }

    if (this.bridge.isInitialized()) {
      this.bridge.module?.EnableCollaboration(false, '');
    }
  }
}

// Main O3DE workflow bridge class
export class O3DEWorkflowBridge extends EventEmitter {
  private module: O3DEModule | null = null;
  private isInitialized: boolean = false;
  private canvas: HTMLCanvasElement;
  private performanceMonitor: PerformanceMonitor;
  private collaborationManager: CollaborationManager;
  private currentWorkflow: Workflow | null = null;
  private currentMode: VisualizationMode = VisualizationMode.ThreeD;
  private currentQuality: QualityLevel = QualityLevel.Medium;
  private userId: string = '';
  private sessionId: string = '';

  constructor(canvasId: string, userId: string = '') {
    super();
    
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    if (!canvas) {
      throw new Error(`Canvas element with id '${canvasId}' not found`);
    }
    
    this.canvas = canvas;
    this.userId = userId;
    this.performanceMonitor = new PerformanceMonitor(this);
    this.collaborationManager = new CollaborationManager(this);
    
    this.setupEventHandlers();
  }

  async initialize(): Promise<void> {
    try {
      console.log('Initializing O3DE Workflow Bridge...');
      
      // Load O3DE WebAssembly module
      const moduleConfig = {
        canvas: this.canvas,
        locateFile: (path: string) => {
          if (path.endsWith('.wasm')) {
            return `/assets/o3de/${path}`;
          }
          return path;
        },
        onRuntimeInitialized: () => {
          this.isInitialized = true;
          this.emit('initialized');
          console.log('O3DE module initialized successfully');
        },
        onAbort: (what: any) => {
          console.error('O3DE module aborted:', what);
          this.emit('error', new Error('O3DE module aborted'));
        }
      };

      // Load the O3DE module (this would be the actual O3DE WebAssembly module)
      // @ts-ignore - O3DE module loading
      this.module = await window.createO3DEModule(moduleConfig);
      
      // Start performance monitoring
      this.performanceMonitor.start();
      
      console.log('O3DE Workflow Bridge initialized successfully');
      
    } catch (error) {
      console.error('Failed to initialize O3DE module:', error);
      this.emit('error', error);
      throw error;
    }
  }

  private setupEventHandlers(): void {
    // Handle canvas resize
    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        this.canvas.width = width;
        this.canvas.height = height;
        this.emit('canvasResized', { width, height });
      }
    });
    resizeObserver.observe(this.canvas);

    // Handle user interactions
    this.canvas.addEventListener('click', this.handleCanvasClick.bind(this));
    this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
    this.canvas.addEventListener('wheel', this.handleMouseWheel.bind(this));
    this.canvas.addEventListener('touchstart', this.handleTouchStart.bind(this));
    this.canvas.addEventListener('touchmove', this.handleTouchMove.bind(this));
    this.canvas.addEventListener('touchend', this.handleTouchEnd.bind(this));

    // Handle keyboard events
    document.addEventListener('keydown', this.handleKeyDown.bind(this));
    document.addEventListener('keyup', this.handleKeyUp.bind(this));
  }

  // Core workflow visualization methods
  async createWorkflowVisualization(workflow: Workflow): Promise<void> {
    if (!this.isInitialized || !this.module) {
      throw new Error('O3DE module not initialized');
    }

    console.log('Creating workflow visualization:', workflow.workflow_id);
    
    this.currentWorkflow = workflow;
    const workflowData = this.transformWorkflowData(workflow);
    
    try {
      this.module.CreateWorkflowVisualization(JSON.stringify(workflowData));
      this.emit('workflowCreated', workflow);
    } catch (error) {
      console.error('Failed to create workflow visualization:', error);
      this.emit('error', error);
    }
  }

  async updateWorkflowVisualization(workflow: Workflow): Promise<void> {
    if (!this.isInitialized || !this.module) {
      throw new Error('O3DE module not initialized');
    }

    console.log('Updating workflow visualization:', workflow.workflow_id);
    
    this.currentWorkflow = workflow;
    const workflowData = this.transformWorkflowData(workflow);
    
    try {
      this.module.UpdateWorkflowVisualization(JSON.stringify(workflowData));
      this.emit('workflowUpdated', workflow);
    } catch (error) {
      console.error('Failed to update workflow visualization:', error);
      this.emit('error', error);
    }
  }

  async updateNodeStatus(nodeId: string, status: NodeStatus): Promise<void> {
    if (!this.isInitialized || !this.module) {
      throw new Error('O3DE module not initialized');
    }

    try {
      this.module.UpdateNodeStatus(nodeId, JSON.stringify(status));
      this.emit('nodeStatusUpdated', { nodeId, status });
    } catch (error) {
      console.error('Failed to update node status:', error);
      this.emit('error', error);
    }
  }

  setVisualizationMode(mode: VisualizationMode): void {
    if (!this.isInitialized || !this.module) {
      throw new Error('O3DE module not initialized');
    }

    this.currentMode = mode;
    this.module.SetVisualizationMode(mode);
    this.emit('visualizationModeChanged', mode);
  }

  setQualityLevel(level: QualityLevel): void {
    if (!this.isInitialized || !this.module) {
      throw new Error('O3DE module not initialized');
    }

    this.currentQuality = level;
    this.module.SetQualityLevel(level);
    this.emit('qualityLevelChanged', level);
  }

  // Collaboration methods
  async enableCollaboration(sessionId: string): Promise<void> {
    this.sessionId = sessionId;
    await this.collaborationManager.initialize(sessionId, this.userId);
    this.emit('collaborationEnabled', sessionId);
  }

  disableCollaboration(): void {
    this.collaborationManager.disable();
    this.sessionId = '';
    this.emit('collaborationDisabled');
  }

  addCollaborativeUser(user: UserPresence): void {
    if (!this.isInitialized || !this.module) {
      throw new Error('O3DE module not initialized');
    }

    this.module.AddCollaborativeUser(JSON.stringify(user));
    this.emit('collaborativeUserAdded', user);
  }

  removeCollaborativeUser(userId: string): void {
    if (!this.isInitialized || !this.module) {
      throw new Error('O3DE module not initialized');
    }

    this.module.RemoveCollaborativeUser(userId);
    this.emit('collaborativeUserRemoved', userId);
  }

  // Layout optimization
  async optimizeLayout(params: LayoutOptimizationParams): Promise<OptimizedLayout> {
    if (!this.isInitialized || !this.module) {
      throw new Error('O3DE module not initialized');
    }

    try {
      const resultJson = this.module.OptimizeLayout(JSON.stringify(params));
      const result: OptimizedLayout = JSON.parse(resultJson);
      this.emit('layoutOptimized', result);
      return result;
    } catch (error) {
      console.error('Failed to optimize layout:', error);
      this.emit('error', error);
      throw error;
    }
  }

  // Performance monitoring
  getPerformanceMetrics(): PerformanceMetrics {
    return this.performanceMonitor.getMetrics();
  }

  // Utility methods
  isInitialized(): boolean {
    return this.isInitialized;
  }

  getCurrentWorkflow(): Workflow | null {
    return this.currentWorkflow;
  }

  getCurrentMode(): VisualizationMode {
    return this.currentMode;
  }

  getCurrentQuality(): QualityLevel {
    return this.currentQuality;
  }

  getCollaborationManager(): CollaborationManager {
    return this.collaborationManager;
  }

  // Event handlers
  private handleCanvasClick(event: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const interaction: UserInteraction = {
      interactionId: `click_${Date.now()}`,
      userId: this.userId,
      interactionType: 'click',
      position: { x, y, z: 0 },
      direction: { x: 0, y: 0, z: -1 },
      targetNodeId: '',
      parameters: {},
      timestamp: Date.now()
    };

    this.handleUserInteraction(interaction);
  }

  private handleMouseMove(event: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const interaction: UserInteraction = {
      interactionId: `hover_${Date.now()}`,
      userId: this.userId,
      interactionType: 'hover',
      position: { x, y, z: 0 },
      direction: { x: 0, y: 0, z: -1 },
      targetNodeId: '',
      parameters: {},
      timestamp: Date.now()
    };

    this.handleUserInteraction(interaction);
    
    // Broadcast cursor movement for collaboration
    if (this.collaborationManager.isEnabled()) {
      this.collaborationManager.broadcastCursorMove({ x, y, z: 0 });
    }
  }

  private handleMouseWheel(event: WheelEvent): void {
    event.preventDefault();
    
    const interaction: UserInteraction = {
      interactionId: `scroll_${Date.now()}`,
      userId: this.userId,
      interactionType: 'scroll',
      position: { x: 0, y: 0, z: 0 },
      direction: { x: 0, y: 0, z: event.deltaY > 0 ? 1 : -1 },
      targetNodeId: '',
      parameters: { deltaY: event.deltaY.toString() },
      timestamp: Date.now()
    };

    this.handleUserInteraction(interaction);
  }

  private handleTouchStart(event: TouchEvent): void {
    event.preventDefault();
    
    const touch = event.touches[0];
    const rect = this.canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    
    const interaction: UserInteraction = {
      interactionId: `touch_start_${Date.now()}`,
      userId: this.userId,
      interactionType: 'touch_start',
      position: { x, y, z: 0 },
      direction: { x: 0, y: 0, z: -1 },
      targetNodeId: '',
      parameters: {},
      timestamp: Date.now()
    };

    this.handleUserInteraction(interaction);
  }

  private handleTouchMove(event: TouchEvent): void {
    event.preventDefault();
    
    const touch = event.touches[0];
    const rect = this.canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    
    const interaction: UserInteraction = {
      interactionId: `touch_move_${Date.now()}`,
      userId: this.userId,
      interactionType: 'touch_move',
      position: { x, y, z: 0 },
      direction: { x: 0, y: 0, z: -1 },
      targetNodeId: '',
      parameters: {},
      timestamp: Date.now()
    };

    this.handleUserInteraction(interaction);
  }

  private handleTouchEnd(event: TouchEvent): void {
    event.preventDefault();
    
    const interaction: UserInteraction = {
      interactionId: `touch_end_${Date.now()}`,
      userId: this.userId,
      interactionType: 'touch_end',
      position: { x: 0, y: 0, z: 0 },
      direction: { x: 0, y: 0, z: -1 },
      targetNodeId: '',
      parameters: {},
      timestamp: Date.now()
    };

    this.handleUserInteraction(interaction);
  }

  private handleKeyDown(event: KeyboardEvent): void {
    // Handle keyboard shortcuts for accessibility
    const interaction: UserInteraction = {
      interactionId: `key_down_${Date.now()}`,
      userId: this.userId,
      interactionType: 'key_down',
      position: { x: 0, y: 0, z: 0 },
      direction: { x: 0, y: 0, z: 0 },
      targetNodeId: '',
      parameters: { key: event.key, code: event.code },
      timestamp: Date.now()
    };

    this.handleUserInteraction(interaction);
  }

  private handleKeyUp(event: KeyboardEvent): void {
    const interaction: UserInteraction = {
      interactionId: `key_up_${Date.now()}`,
      userId: this.userId,
      interactionType: 'key_up',
      position: { x: 0, y: 0, z: 0 },
      direction: { x: 0, y: 0, z: 0 },
      targetNodeId: '',
      parameters: { key: event.key, code: event.code },
      timestamp: Date.now()
    };

    this.handleUserInteraction(interaction);
  }

  private handleUserInteraction(interaction: UserInteraction): void {
    if (!this.isInitialized || !this.module) {
      return;
    }

    try {
      this.module.HandleUserInteraction(JSON.stringify(interaction));
      this.emit('userInteraction', interaction);
      
      // Broadcast interaction for collaboration
      if (this.collaborationManager.isEnabled()) {
        this.collaborationManager.broadcastInteraction(interaction);
      }
    } catch (error) {
      console.error('Failed to handle user interaction:', error);
      this.emit('error', error);
    }
  }

  // Helper methods
  private transformWorkflowData(workflow: Workflow): any {
    return {
      workflowId: workflow.workflow_id,
      metadata: {
        name: workflow.name || workflow.workflow_id,
        description: workflow.description || '',
        status: workflow.status || 'active',
        createdAt: workflow.created_at || new Date().toISOString(),
        updatedAt: workflow.updated_at || new Date().toISOString()
      },
      nodes: workflow.tasks.map(task => ({
        nodeId: task.task_id,
        type: this.getNodeType(task),
        label: task.message || task.task_id,
        description: task.description || '',
        status: task.status || 'pending',
        progress: task.progress || 0,
        dependencies: task.dependencies || [],
        position: this.calculateNodePosition(task),
        visualProperties: this.getNodeVisualProperties(task)
      })),
      edges: this.calculateEdges(workflow.tasks)
    };
  }

  private getNodeType(task: TaskBlock): number {
    // Map task types to O3DE node types
    switch (task.type) {
      case 'approval': return 1;
      case 'conditional': return 2;
      case 'loop': return 3;
      case 'parallel': return 4;
      case 'agent': return 7;
      case 'data': return 8;
      case 'human': return 10;
      default: return 0; // Task
    }
  }

  private calculateNodePosition(task: TaskBlock): Vector3 {
    // Simple positioning logic - would be enhanced by AI layout optimization
    const baseX = Math.random() * 20 - 10;
    const baseY = Math.random() * 20 - 10;
    const baseZ = (task.dependencies?.length || 0) * 2;
    
    return { x: baseX, y: baseY, z: baseZ };
  }

  private getNodeVisualProperties(task: TaskBlock): any {
    return {
      color: this.getStatusColor(task.status),
      scale: { x: 1, y: 1, z: 1 },
      opacity: 1.0,
      showLabel: true,
      showProgressBar: task.progress && task.progress > 0,
      labelText: task.message || task.task_id
    };
  }

  private getStatusColor(status: string): { r: number; g: number; b: number; a: number } {
    switch (status) {
      case 'pending': return { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };
      case 'running': return { r: 0.0, g: 0.5, b: 1.0, a: 1.0 };
      case 'completed': return { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
      case 'failed': return { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
      case 'pending_approval': return { r: 1.0, g: 1.0, b: 0.0, a: 1.0 };
      default: return { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };
    }
  }

  private calculateEdges(tasks: TaskBlock[]): any[] {
    const edges: any[] = [];
    
    tasks.forEach(task => {
      if (task.dependencies) {
        task.dependencies.forEach(depId => {
          edges.push({
            edgeId: `edge_${depId}_${task.task_id}`,
            sourceNodeId: depId,
            targetNodeId: task.task_id,
            type: 0, // Sequential
            label: '',
            visualProperties: {
              color: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },
              thickness: 0.1,
              animated: false
            }
          });
        });
      }
    });
    
    return edges;
  }

  // Cleanup
  dispose(): void {
    this.performanceMonitor.stop();
    this.collaborationManager.disable();
    
    if (this.module) {
      // Cleanup O3DE resources
      this.module = null;
    }
    
    this.isInitialized = false;
    this.removeAllListeners();
  }
}

export default O3DEWorkflowBridge; 