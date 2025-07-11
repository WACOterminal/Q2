#pragma once

#include <AzCore/Component/Component.h>
#include <AzCore/Component/TickBus.h>
#include <AzCore/Math/Transform.h>
#include <AzCore/std/containers/unordered_map.h>
#include <AzCore/std/containers/vector.h>
#include <AzCore/std/string/string.h>
#include <AzCore/std/smart_ptr/unique_ptr.h>
#include <AzCore/Memory/SystemAllocator.h>
#include <AzCore/RTTI/RTTI.h>
#include <AzCore/Serialization/SerializeContext.h>
#include <AzFramework/Entity/EntityContextBus.h>
#include <AzFramework/Entity/GameEntityContextBus.h>

#include "WorkflowVisualization3DBus.h"
#include "WorkflowData.h"
#include "NodeFactory.h"
#include "CollaborationManager.h"
#include "PerformanceMonitor.h"
#include "InteractionSystem.h"
#include "AILayoutOptimizer.h"
#include "SpatialAudioManager.h"
#include "AccessibilityManager.h"

#ifdef EMSCRIPTEN
#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#endif

namespace Q3D
{
    class WorkflowVisualization3DSystemComponent
        : public AZ::Component
        , public WorkflowVisualization3DRequestBus::Handler
        , public AZ::TickBus::Handler
    {
    public:
        AZ_COMPONENT(WorkflowVisualization3DSystemComponent, "{12345678-1234-5678-9ABC-DEF012345678}");

        static void Reflect(AZ::ReflectContext* context);

        static void GetProvidedServices(AZ::ComponentDescriptor::DependencyArrayType& provided);
        static void GetIncompatibleServices(AZ::ComponentDescriptor::DependencyArrayType& incompatible);
        static void GetRequiredServices(AZ::ComponentDescriptor::DependencyArrayType& required);
        static void GetDependentServices(AZ::ComponentDescriptor::DependencyArrayType& dependent);

        WorkflowVisualization3DSystemComponent();
        ~WorkflowVisualization3DSystemComponent();

        // Component overrides
        void Init() override;
        void Activate() override;
        void Deactivate() override;

        // WorkflowVisualization3DRequestBus
        void CreateWorkflowVisualization(const WorkflowData& workflowData) override;
        void UpdateWorkflowVisualization(const WorkflowData& workflowData) override;
        void UpdateNodeStatus(const AZStd::string& nodeId, const NodeStatus& status) override;
        void SetVisualizationMode(VisualizationMode mode) override;
        void SetQualityLevel(QualityLevel level) override;
        void EnableCollaboration(bool enable, const AZStd::string& sessionId) override;
        void AddCollaborativeUser(const UserPresence& user) override;
        void RemoveCollaborativeUser(const AZStd::string& userId) override;
        void UpdateUserCursor(const AZStd::string& userId, const AZ::Vector3& position) override;
        void HandleUserInteraction(const UserInteraction& interaction) override;
        AZStd::string OptimizeLayout(const LayoutOptimizationParams& params) override;
        PerformanceMetrics GetPerformanceMetrics() const override;
        bool IsInitialized() const override;
        
        // Accessibility methods
        void EnableAccessibilityFeature(AccessibilityFeature feature, bool enabled);
        void SetAccessibilityProfile(const AZStd::string& profileId);
        void SetAccessibilityLevel(AccessibilityLevel level);
        void AnnounceToScreenReader(const AZStd::string& message);
        void TriggerHapticFeedback(const AZStd::string& feedbackType, float intensity);
        void UpdateFocusedElement(const AZStd::string& elementId);
        bool ProcessVoiceCommand(const AZStd::string& command);
        void ValidateAccessibilityCompliance();
        AccessibilitySettings GetAccessibilitySettings() const;

        // AZ::TickBus
        void OnTick(float deltaTime, AZ::ScriptTimePoint time) override;

    private:
        // Core visualization components
        AZStd::unique_ptr<NodeFactory> m_nodeFactory;
        AZStd::unique_ptr<CollaborationManager> m_collaborationManager;
        AZStd::unique_ptr<PerformanceMonitor> m_performanceMonitor;
        AZStd::unique_ptr<InteractionSystem> m_interactionSystem;
        AZStd::unique_ptr<AILayoutOptimizer> m_aiLayoutOptimizer;
        AZStd::unique_ptr<SpatialAudioManager> m_spatialAudioManager;
        AZStd::unique_ptr<AccessibilityManager> m_accessibilityManager;

        // State management
        WorkflowData m_currentWorkflow;
        AZStd::unordered_map<AZStd::string, AZ::EntityId> m_nodeEntities;
        AZStd::unordered_map<AZStd::string, AZ::EntityId> m_edgeEntities;
        AZStd::unordered_map<AZStd::string, UserPresence> m_collaborativeUsers;
        
        // Configuration
        VisualizationMode m_currentMode = VisualizationMode::ThreeD;
        QualityLevel m_currentQuality = QualityLevel::Medium;
        bool m_collaborationEnabled = false;
        AZStd::string m_sessionId;
        
        // Performance tracking
        float m_timeSinceLastUpdate = 0.0f;
        float m_targetUpdateRate = 1.0f / 60.0f; // 60 FPS
        bool m_needsRender = false;
        bool m_isInitialized = false;
        
        // WebAssembly interface
        #ifdef EMSCRIPTEN
        void SetupWebAssemblyBindings();
        #endif
        
        // Helper methods
        void InitializeComponents();
        void UpdateVisualization(float deltaTime);
        void CleanupWorkflow();
        void ApplyLayoutOptimization(const OptimizedLayout& layout);
        void UpdateCollaborativeUsers(float deltaTime);
        void ProcessPendingInteractions();
        
        // Event handlers
        void OnWorkflowDataChanged();
        void OnNodeStatusChanged(const AZStd::string& nodeId, const NodeStatus& status);
        void OnCollaborationEvent(const CollaborationEvent& event);
        void OnPerformanceMetricsChanged(const PerformanceMetrics& metrics);
        
        // Layout management
        void CalculateOptimalLayout();
        void ApplyForceDirectedLayout();
        void ApplyHierarchicalLayout();
        void ApplyCircularLayout();
        void ApplyAIOptimizedLayout();
        
        // Rendering optimization
        void OptimizeRenderingPerformance();
        void UpdateLevelOfDetail();
        void CullInvisibleNodes();
        void BatchRenderOperations();
        
        // Memory management
        void CleanupUnusedAssets();
        void OptimizeMemoryUsage();
        
        // Accessibility support
        void UpdateAccessibilityFeatures();
        void GenerateAudioDescriptions();
        void UpdateScreenReaderSupport();
    };

    // WebAssembly C interface functions
    #ifdef EMSCRIPTEN
    extern "C" {
        EMSCRIPTEN_KEEPALIVE void CreateWorkflowVisualization(const char* workflowDataJson);
        EMSCRIPTEN_KEEPALIVE void UpdateNodeStatus(const char* nodeId, const char* statusJson);
        EMSCRIPTEN_KEEPALIVE void HandleUserInteraction(const char* interactionJson);
        EMSCRIPTEN_KEEPALIVE void SetCollaborationMode(bool enabled, const char* sessionId);
        EMSCRIPTEN_KEEPALIVE void AddCollaborativeUser(const char* userPresenceJson);
        EMSCRIPTEN_KEEPALIVE void RemoveCollaborativeUser(const char* userId);
        EMSCRIPTEN_KEEPALIVE void UpdateUserCursor(const char* userId, float x, float y, float z);
        EMSCRIPTEN_KEEPALIVE void SetVisualizationMode(int mode);
        EMSCRIPTEN_KEEPALIVE void SetQualityLevel(int level);
        EMSCRIPTEN_KEEPALIVE const char* OptimizeLayout(const char* paramsJson);
        EMSCRIPTEN_KEEPALIVE const char* GetPerformanceMetrics();
        EMSCRIPTEN_KEEPALIVE bool IsSystemInitialized();
    }
    #endif
} // namespace Q3D 