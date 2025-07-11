#pragma once

#include <AzCore/EBus/EBus.h>
#include <AzCore/Interface/Interface.h>
#include <AzCore/Math/Vector3.h>
#include <AzCore/Math/Color.h>
#include <AzCore/std/string/string.h>
#include <AzCore/std/containers/vector.h>
#include <AzCore/std/containers/unordered_map.h>

#include "WorkflowData.h"

namespace Q3D
{
    enum class VisualizationMode : int
    {
        TwoD = 0,
        ThreeD = 1,
        VR = 2,
        AR = 3
    };

    enum class QualityLevel : int
    {
        Low = 0,
        Medium = 1,
        High = 2,
        Ultra = 3
    };

    enum class AccessibilityFeature : int
    {
        ScreenReader = 0,
        KeyboardNavigation = 1,
        VoiceCommands = 2,
        HapticFeedback = 3,
        HighContrast = 4,
        ColorBlindSupport = 5,
        MotionReduction = 6,
        FontScaling = 7,
        FocusIndicators = 8,
        SkipLinks = 9,
        LiveRegions = 10,
        AudioDescriptions = 11,
        CaptionsSubtitles = 12,
        GestureAlternatives = 13,
        SlowMotionMode = 14,
        SimplifiedInterface = 15,
        CognitiveAids = 16,
        CustomControls = 17,
        BrailleSupport = 18,
        EyeTracking = 19
    };

    enum class AccessibilityLevel : int
    {
        None = 0,
        Basic = 1,
        Enhanced = 2,
        Full = 3,
        Advanced = 4
    };

    struct AccessibilitySettings
    {
        AccessibilityLevel level = AccessibilityLevel::Basic;
        AZStd::string activeProfileId;
        AZStd::unordered_map<AccessibilityFeature, bool> enabledFeatures;
        AZStd::unordered_map<AZStd::string, AZStd::string> customSettings;
        bool enableGlobalShortcuts = true;
        bool enableAutoDetection = true;
        bool enableTelemetry = false;
        bool enableDebugMode = false;
    };

    enum class NodeStatusType : int
    {
        Pending = 0,
        Running = 1,
        Completed = 2,
        Failed = 3,
        PendingApproval = 4,
        Cancelled = 5
    };

    struct NodeStatus
    {
        NodeStatusType status = NodeStatusType::Pending;
        float progress = 0.0f;
        AZStd::string message;
        AZStd::string errorDetails;
        AZStd::unordered_map<AZStd::string, AZStd::string> metadata;
    };

    struct UserPresence
    {
        AZStd::string userId;
        AZStd::string displayName;
        AZ::Color avatarColor;
        AZ::Vector3 position;
        AZ::Vector3 orientation;
        AZStd::string selectedNodeId;
        bool isEditing = false;
        bool voiceActive = false;
        float lastActivity = 0.0f;
        AZStd::unordered_map<AZStd::string, AZStd::string> metadata;
    };

    struct UserInteraction
    {
        AZStd::string interactionId;
        AZStd::string userId;
        AZStd::string interactionType; // "click", "drag", "hover", "voice", "gesture"
        AZ::Vector3 position;
        AZ::Vector3 direction;
        AZStd::string targetNodeId;
        AZStd::unordered_map<AZStd::string, AZStd::string> parameters;
        float timestamp = 0.0f;
    };

    struct CollaborationEvent
    {
        AZStd::string eventId;
        AZStd::string eventType; // "user_join", "user_leave", "node_edit", "comment_add", etc.
        AZStd::string userId;
        AZStd::string sessionId;
        AZStd::unordered_map<AZStd::string, AZStd::string> data;
        float timestamp = 0.0f;
    };

    struct PerformanceMetrics
    {
        float frameTime = 0.0f;
        float renderTime = 0.0f;
        float updateTime = 0.0f;
        int drawCalls = 0;
        int vertices = 0;
        int triangles = 0;
        float memoryUsage = 0.0f;
        float gpuMemoryUsage = 0.0f;
        int activeNodes = 0;
        int visibleNodes = 0;
        QualityLevel currentQuality = QualityLevel::Medium;
        float cpuUsage = 0.0f;
        float gpuUsage = 0.0f;
        float networkLatency = 0.0f;
        AZStd::unordered_map<AZStd::string, float> customMetrics;
    };

    struct LayoutOptimizationParams
    {
        WorkflowData workflow;
        AZ::Vector3 canvasSize;
        AZStd::string layoutAlgorithm; // "force_directed", "hierarchical", "circular", "ai_optimized"
        AZStd::unordered_map<AZStd::string, AZStd::string> userPreferences;
        PerformanceMetrics performanceConstraints;
        bool enableCollaboration = false;
        int maxIterations = 1000;
        float convergenceThreshold = 0.001f;
    };

    struct OptimizedLayout
    {
        AZStd::string layoutId;
        AZStd::string algorithm;
        AZStd::unordered_map<AZStd::string, AZ::Vector3> nodePositions;
        AZStd::unordered_map<AZStd::string, AZStd::vector<AZ::Vector3>> edgePaths;
        AZ::Vector3 cameraPosition;
        AZ::Vector3 cameraTarget;
        QualityLevel recommendedQuality;
        float optimizationScore = 0.0f;
        AZStd::unordered_map<AZStd::string, float> layoutMetrics;
    };

    // Main interface for workflow visualization requests
    class WorkflowVisualization3DRequests
    {
    public:
        AZ_RTTI(WorkflowVisualization3DRequests, "{87654321-4321-8765-4321-876543210987}");
        virtual ~WorkflowVisualization3DRequests() = default;

        // Core visualization methods
        virtual void CreateWorkflowVisualization(const WorkflowData& workflowData) = 0;
        virtual void UpdateWorkflowVisualization(const WorkflowData& workflowData) = 0;
        virtual void UpdateNodeStatus(const AZStd::string& nodeId, const NodeStatus& status) = 0;
        virtual void SetVisualizationMode(VisualizationMode mode) = 0;
        virtual void SetQualityLevel(QualityLevel level) = 0;

        // Collaboration methods
        virtual void EnableCollaboration(bool enable, const AZStd::string& sessionId) = 0;
        virtual void AddCollaborativeUser(const UserPresence& user) = 0;
        virtual void RemoveCollaborativeUser(const AZStd::string& userId) = 0;
        virtual void UpdateUserCursor(const AZStd::string& userId, const AZ::Vector3& position) = 0;
        virtual void HandleUserInteraction(const UserInteraction& interaction) = 0;

        // AI optimization methods
        virtual AZStd::string OptimizeLayout(const LayoutOptimizationParams& params) = 0;

        // Performance and monitoring
        virtual PerformanceMetrics GetPerformanceMetrics() const = 0;
        virtual bool IsInitialized() const = 0;

        // Accessibility methods
        virtual void EnableAccessibilityFeature(AccessibilityFeature feature, bool enabled) = 0;
        virtual void SetAccessibilityProfile(const AZStd::string& profileId) = 0;
        virtual void SetAccessibilityLevel(AccessibilityLevel level) = 0;
        virtual void AnnounceToScreenReader(const AZStd::string& message) = 0;
        virtual void TriggerHapticFeedback(const AZStd::string& feedbackType, float intensity) = 0;
        virtual void UpdateFocusedElement(const AZStd::string& elementId) = 0;
        virtual bool ProcessVoiceCommand(const AZStd::string& command) = 0;
        virtual void ValidateAccessibilityCompliance() = 0;
        virtual AccessibilitySettings GetAccessibilitySettings() const = 0;
    };

    class WorkflowVisualization3DRequestBus
        : public AZ::EBusTraits
    {
    public:
        // EBusTraits overrides
        static const AZ::EBusHandlerPolicy HandlerPolicy = AZ::EBusHandlerPolicy::Single;
        static const AZ::EBusAddressPolicy AddressPolicy = AZ::EBusAddressPolicy::Single;
        using MutexType = AZStd::recursive_mutex;

        using Bus = AZ::EBus<WorkflowVisualization3DRequests>;
    };

    using WorkflowVisualization3DRequestBus = AZ::EBus<WorkflowVisualization3DRequests>;

    // Notification interface for workflow visualization events
    class WorkflowVisualization3DNotifications
    {
    public:
        AZ_RTTI(WorkflowVisualization3DNotifications, "{13579246-8024-6813-5792-468135792468}");
        virtual ~WorkflowVisualization3DNotifications() = default;

        // Visualization events
        virtual void OnWorkflowVisualizationCreated(const AZStd::string& workflowId) {}
        virtual void OnWorkflowVisualizationUpdated(const AZStd::string& workflowId) {}
        virtual void OnVisualizationModeChanged(VisualizationMode mode) {}
        virtual void OnQualityLevelChanged(QualityLevel level) {}

        // Node events
        virtual void OnNodeStatusChanged(const AZStd::string& nodeId, const NodeStatus& status) {}
        virtual void OnNodeSelected(const AZStd::string& nodeId, const AZStd::string& userId) {}
        virtual void OnNodeInteraction(const AZStd::string& nodeId, const UserInteraction& interaction) {}

        // Collaboration events
        virtual void OnCollaborationEnabled(const AZStd::string& sessionId) {}
        virtual void OnCollaborationDisabled() {}
        virtual void OnUserJoined(const UserPresence& user) {}
        virtual void OnUserLeft(const AZStd::string& userId) {}
        virtual void OnCollaborationEvent(const CollaborationEvent& event) {}

        // Performance events
        virtual void OnPerformanceMetricsChanged(const PerformanceMetrics& metrics) {}
        virtual void OnQualityAutoAdjusted(QualityLevel oldLevel, QualityLevel newLevel) {}

        // Layout events
        virtual void OnLayoutOptimizationStarted(const AZStd::string& layoutId) {}
        virtual void OnLayoutOptimizationCompleted(const AZStd::string& layoutId, const OptimizedLayout& layout) {}
        virtual void OnLayoutOptimizationFailed(const AZStd::string& layoutId, const AZStd::string& error) {}

        // Error events
        virtual void OnVisualizationError(const AZStd::string& error) {}
        virtual void OnCollaborationError(const AZStd::string& error) {}
        virtual void OnPerformanceWarning(const AZStd::string& warning) {}

        // Accessibility events
        virtual void OnAccessibilityFeatureEnabled(AccessibilityFeature feature, bool enabled) {}
        virtual void OnAccessibilityProfileChanged(const AZStd::string& profileId) {}
        virtual void OnAccessibilityLevelChanged(AccessibilityLevel level) {}
        virtual void OnScreenReaderAnnouncement(const AZStd::string& message) {}
        virtual void OnHapticFeedbackTriggered(const AZStd::string& feedbackType, float intensity) {}
        virtual void OnFocusedElementChanged(const AZStd::string& elementId) {}
        virtual void OnVoiceCommandProcessed(const AZStd::string& command, bool success) {}
        virtual void OnAccessibilityValidationCompleted(bool isCompliant) {}
        virtual void OnAccessibilitySettingsChanged(const AccessibilitySettings& settings) {}
    };

    class WorkflowVisualization3DNotificationBus
        : public AZ::EBusTraits
    {
    public:
        // EBusTraits overrides
        static const AZ::EBusHandlerPolicy HandlerPolicy = AZ::EBusHandlerPolicy::Multiple;
        static const AZ::EBusAddressPolicy AddressPolicy = AZ::EBusAddressPolicy::Single;
        using MutexType = AZStd::recursive_mutex;

        using Bus = AZ::EBus<WorkflowVisualization3DNotifications>;
    };

    using WorkflowVisualization3DNotificationBus = AZ::EBus<WorkflowVisualization3DNotifications>;

} // namespace Q3D 