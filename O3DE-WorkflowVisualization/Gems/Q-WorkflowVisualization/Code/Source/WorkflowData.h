#pragma once

#include <AzCore/std/string/string.h>
#include <AzCore/std/containers/vector.h>
#include <AzCore/std/containers/unordered_map.h>
#include <AzCore/Math/Vector3.h>
#include <AzCore/Math/Color.h>
#include <AzCore/RTTI/RTTI.h>
#include <AzCore/Serialization/SerializeContext.h>

namespace Q3D
{
    enum class NodeType : int
    {
        Task = 0,
        Approval = 1,
        Conditional = 2,
        Loop = 3,
        Parallel = 4,
        Start = 5,
        End = 6,
        Agent = 7,
        DataProcessing = 8,
        Integration = 9,
        Human = 10
    };

    enum class EdgeType : int
    {
        Sequential = 0,
        Conditional = 1,
        DataFlow = 2,
        Dependency = 3,
        Parallel = 4,
        Loop = 5,
        Error = 6
    };

    struct NodeVisualizationProperties
    {
        AZ::Vector3 position = AZ::Vector3::CreateZero();
        AZ::Vector3 scale = AZ::Vector3::CreateOne();
        AZ::Color color = AZ::Color::CreateOne();
        float opacity = 1.0f;
        AZStd::string meshAsset;
        AZStd::string materialAsset;
        AZStd::string iconAsset;
        bool visible = true;
        bool interactable = true;
        float interactionRadius = 1.0f;
        
        // Animation properties
        bool hasIdleAnimation = false;
        bool hasExecutionAnimation = false;
        bool hasCompletionAnimation = false;
        float animationSpeed = 1.0f;
        
        // Visual effects
        bool hasParticleEffects = false;
        bool hasGlowEffect = false;
        bool hasLightSource = false;
        bool castsShadows = true;
        
        // UI properties
        bool showLabel = true;
        bool showProgressBar = false;
        bool showStatusIcon = true;
        AZStd::string labelText;
        
        // Accessibility
        AZStd::string accessibilityLabel;
        AZStd::string audioDescription;
        bool isAccessibilityFocusable = true;
    };

    struct EdgeVisualizationProperties
    {
        AZ::Color color = AZ::Color::CreateOne();
        float thickness = 0.1f;
        float opacity = 1.0f;
        bool animated = false;
        float animationSpeed = 1.0f;
        AZStd::string materialAsset;
        
        // Path properties
        bool curved = true;
        float curvature = 0.5f;
        AZStd::vector<AZ::Vector3> controlPoints;
        
        // Visual effects
        bool hasParticleFlow = false;
        bool hasGlowEffect = false;
        bool showArrowHead = true;
        bool showLabel = false;
        AZStd::string labelText;
        
        // Interaction
        bool isInteractable = false;
        float interactionRadius = 0.5f;
    };

    struct WorkflowNode
    {
        AZStd::string nodeId;
        NodeType type = NodeType::Task;
        AZStd::string label;
        AZStd::string description;
        AZStd::string message;
        AZStd::vector<AZStd::string> dependencies;
        AZStd::unordered_map<AZStd::string, AZStd::string> metadata;
        NodeVisualizationProperties visualProperties;
        
        // Task-specific properties
        AZStd::string taskType;
        AZStd::string agentPersonality;
        AZStd::string prompt;
        AZStd::vector<AZStd::string> tools;
        int maxRetries = 3;
        float timeoutSeconds = 300.0f;
        
        // Conditional properties
        AZStd::string condition;
        AZStd::vector<AZStd::string> trueBranch;
        AZStd::vector<AZStd::string> falseBranch;
        
        // Loop properties
        AZStd::string loopCondition;
        int maxIterations = 10;
        AZStd::vector<AZStd::string> loopBody;
        
        // Parallel properties
        AZStd::vector<AZStd::vector<AZStd::string>> parallelBranches;
        bool waitForAll = true;
        
        // Approval properties
        AZStd::string approvalMessage;
        AZStd::vector<AZStd::string> approvers;
        bool requiresAllApprovals = false;
        
        // Status tracking
        AZStd::string status = "pending";
        float progress = 0.0f;
        AZStd::string currentMessage;
        AZStd::string errorMessage;
        float startTime = 0.0f;
        float endTime = 0.0f;
        
        // Collaboration
        AZStd::string currentEditor;
        AZStd::vector<AZStd::string> comments;
        bool isLocked = false;
        
        // Reflection support
        AZ_TYPE_INFO(WorkflowNode, "{11111111-2222-3333-4444-555555555555}");
        static void Reflect(AZ::ReflectContext* context);
    };

    struct WorkflowEdge
    {
        AZStd::string edgeId;
        AZStd::string sourceNodeId;
        AZStd::string targetNodeId;
        EdgeType type = EdgeType::Sequential;
        AZStd::string label;
        AZStd::string condition;
        AZStd::unordered_map<AZStd::string, AZStd::string> metadata;
        EdgeVisualizationProperties visualProperties;
        
        // Data flow properties
        AZStd::string dataSchema;
        AZStd::string dataTransform;
        bool isDataRequired = false;
        
        // Conditional properties
        AZStd::string conditionExpression;
        AZStd::string conditionResult; // "true", "false", "unknown"
        
        // Error handling
        AZStd::string errorCondition;
        AZStd::string errorAction; // "retry", "skip", "fail", "redirect"
        AZStd::string errorTargetNodeId;
        
        // Performance
        float executionTime = 0.0f;
        int executionCount = 0;
        float averageLatency = 0.0f;
        
        // Reflection support
        AZ_TYPE_INFO(WorkflowEdge, "{22222222-3333-4444-5555-666666666666}");
        static void Reflect(AZ::ReflectContext* context);
    };

    struct WorkflowMetadata
    {
        AZStd::string workflowId;
        AZStd::string name;
        AZStd::string description;
        AZStd::string version;
        AZStd::string author;
        AZStd::string createdAt;
        AZStd::string updatedAt;
        AZStd::vector<AZStd::string> tags;
        AZStd::unordered_map<AZStd::string, AZStd::string> customMetadata;
        
        // Execution properties
        AZStd::string status = "draft"; // "draft", "active", "completed", "failed", "paused"
        float executionProgress = 0.0f;
        AZStd::string currentNodeId;
        float startTime = 0.0f;
        float endTime = 0.0f;
        int totalNodes = 0;
        int completedNodes = 0;
        int failedNodes = 0;
        
        // Collaboration
        AZStd::string sessionId;
        AZStd::vector<AZStd::string> collaborators;
        bool isCollaborative = false;
        
        // Visualization preferences
        AZStd::string preferredLayout = "auto";
        AZStd::string visualizationMode = "3d";
        AZStd::string qualityLevel = "medium";
        bool showAnimations = true;
        bool showParticleEffects = true;
        bool enableSpatialAudio = true;
        
        // Performance tracking
        float averageExecutionTime = 0.0f;
        int totalExecutions = 0;
        float successRate = 0.0f;
        AZStd::unordered_map<AZStd::string, float> performanceMetrics;
        
        // Reflection support
        AZ_TYPE_INFO(WorkflowMetadata, "{33333333-4444-5555-6666-777777777777}");
        static void Reflect(AZ::ReflectContext* context);
    };

    struct WorkflowData
    {
        WorkflowMetadata metadata;
        AZStd::vector<WorkflowNode> nodes;
        AZStd::vector<WorkflowEdge> edges;
        
        // Layout information
        AZStd::unordered_map<AZStd::string, AZ::Vector3> nodePositions;
        AZStd::unordered_map<AZStd::string, AZStd::vector<AZ::Vector3>> edgePaths;
        AZ::Vector3 cameraPosition = AZ::Vector3(0.0f, 0.0f, 10.0f);
        AZ::Vector3 cameraTarget = AZ::Vector3::CreateZero();
        float cameraFOV = 60.0f;
        
        // Bounding information
        AZ::Vector3 boundingBoxMin = AZ::Vector3::CreateZero();
        AZ::Vector3 boundingBoxMax = AZ::Vector3::CreateZero();
        AZ::Vector3 center = AZ::Vector3::CreateZero();
        float radius = 0.0f;
        
        // Validation
        bool isValid = true;
        AZStd::vector<AZStd::string> validationErrors;
        
        // Helper methods
        WorkflowNode* FindNodeById(const AZStd::string& nodeId);
        const WorkflowNode* FindNodeById(const AZStd::string& nodeId) const;
        WorkflowEdge* FindEdgeById(const AZStd::string& edgeId);
        const WorkflowEdge* FindEdgeById(const AZStd::string& edgeId) const;
        
        AZStd::vector<WorkflowNode*> GetDependentNodes(const AZStd::string& nodeId);
        AZStd::vector<WorkflowNode*> GetDependencyNodes(const AZStd::string& nodeId);
        AZStd::vector<WorkflowEdge*> GetIncomingEdges(const AZStd::string& nodeId);
        AZStd::vector<WorkflowEdge*> GetOutgoingEdges(const AZStd::string& nodeId);
        
        bool ValidateWorkflow();
        void CalculateBoundingBox();
        void OptimizeLayout();
        void UpdateProgress();
        
        // Serialization
        AZStd::string ToJson() const;
        bool FromJson(const AZStd::string& json);
        
        // Reflection support
        AZ_TYPE_INFO(WorkflowData, "{44444444-5555-6666-7777-888888888888}");
        static void Reflect(AZ::ReflectContext* context);
    };

    // Utility functions
    AZStd::string NodeTypeToString(NodeType type);
    NodeType StringToNodeType(const AZStd::string& str);
    AZStd::string EdgeTypeToString(EdgeType type);
    EdgeType StringToEdgeType(const AZStd::string& str);
    
    // JSON serialization helpers
    AZStd::string SerializeWorkflowData(const WorkflowData& data);
    WorkflowData DeserializeWorkflowData(const AZStd::string& json);
    
    // Validation functions
    bool ValidateNodeData(const WorkflowNode& node, AZStd::vector<AZStd::string>& errors);
    bool ValidateEdgeData(const WorkflowEdge& edge, AZStd::vector<AZStd::string>& errors);
    bool ValidateWorkflowData(const WorkflowData& data, AZStd::vector<AZStd::string>& errors);
    
    // Layout calculation functions
    void CalculateForceDirectedLayout(WorkflowData& data, int iterations = 1000);
    void CalculateHierarchicalLayout(WorkflowData& data);
    void CalculateCircularLayout(WorkflowData& data);
    void CalculateGridLayout(WorkflowData& data);
    
    // Complexity analysis
    struct WorkflowComplexity
    {
        int nodeCount = 0;
        int edgeCount = 0;
        int maxDepth = 0;
        int parallelBranches = 0;
        int conditionalNodes = 0;
        int loopNodes = 0;
        float complexityScore = 0.0f;
        
        AZ_TYPE_INFO(WorkflowComplexity, "{55555555-6666-7777-8888-999999999999}");
        static void Reflect(AZ::ReflectContext* context);
    };
    
    WorkflowComplexity AnalyzeWorkflowComplexity(const WorkflowData& data);
    
} // namespace Q3D 