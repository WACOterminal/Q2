#pragma once

#include <AzCore/Component/Entity.h>
#include <AzCore/Math/Vector3.h>
#include <AzCore/Math/Color.h>
#include <AzCore/std/containers/unordered_map.h>
#include <AzCore/std/string/string.h>
#include <AzCore/std/smart_ptr/unique_ptr.h>
#include <AzFramework/Components/TransformComponent.h>
#include <Atom/RPI.Public/Model/ModelAsset.h>
#include <Atom/RPI.Public/Material/MaterialAsset.h>

#include "WorkflowData.h"
#include "WorkflowVisualization3DBus.h"

namespace Q3D
{
    // Forward declarations
    class NodeAnimationController;
    class NodeParticleEffects;
    class NodeInteractionHandler;
    class NodeAccessibilityFeatures;

    struct NodeVisualizationSpec
    {
        NodeType type;
        AZ::Vector3 position;
        AZ::Vector3 scale = AZ::Vector3::CreateOne();
        AZ::Color baseColor = AZ::Color::CreateOne();
        AZStd::string meshAsset;
        AZStd::string materialAsset;
        AZStd::string iconAsset;
        
        // Animation properties
        bool hasIdleAnimation = false;
        bool hasExecutionAnimation = false;
        bool hasCompletionAnimation = false;
        float animationSpeed = 1.0f;
        
        // Interaction properties
        bool isInteractable = true;
        float interactionRadius = 1.0f;
        
        // Visual effects
        bool hasParticleEffects = false;
        bool hasLighting = false;
        bool castsShadows = true;
        bool hasGlowEffect = false;
        
        // UI properties
        bool showLabel = true;
        bool showProgressBar = false;
        bool showStatusIcon = true;
        AZStd::string labelText;
        
        // Accessibility
        AZStd::string accessibilityLabel;
        AZStd::string audioDescription;
        bool isAccessibilityFocusable = true;
        
        // Performance
        QualityLevel lodLevel = QualityLevel::Medium;
        float renderDistance = 100.0f;
        bool enableOcclusion = true;
    };

    class NodeFactory
    {
    public:
        NodeFactory();
        ~NodeFactory();

        // Core node creation methods
        AZ::EntityId CreateNode(const WorkflowNode& nodeData);
        AZ::EntityId CreateNode(const NodeVisualizationSpec& spec);
        void UpdateNode(AZ::EntityId nodeId, const WorkflowNode& nodeData);
        void UpdateNodeStatus(AZ::EntityId nodeId, const NodeStatus& status);
        void DestroyNode(AZ::EntityId nodeId);
        
        // Node type specific creation
        AZ::EntityId CreateTaskNode(const WorkflowNode& nodeData);
        AZ::EntityId CreateApprovalNode(const WorkflowNode& nodeData);
        AZ::EntityId CreateConditionalNode(const WorkflowNode& nodeData);
        AZ::EntityId CreateLoopNode(const WorkflowNode& nodeData);
        AZ::EntityId CreateParallelNode(const WorkflowNode& nodeData);
        AZ::EntityId CreateAgentNode(const WorkflowNode& nodeData);
        AZ::EntityId CreateDataProcessingNode(const WorkflowNode& nodeData);
        AZ::EntityId CreateHumanNode(const WorkflowNode& nodeData);
        
        // Visual updates
        void UpdateNodeColor(AZ::EntityId nodeId, const AZ::Color& color);
        void UpdateNodeScale(AZ::EntityId nodeId, const AZ::Vector3& scale);
        void UpdateNodePosition(AZ::EntityId nodeId, const AZ::Vector3& position);
        void UpdateNodeOpacity(AZ::EntityId nodeId, float opacity);
        void UpdateNodeLabel(AZ::EntityId nodeId, const AZStd::string& label);
        void UpdateNodeProgress(AZ::EntityId nodeId, float progress);
        
        // Animation control
        void PlayNodeAnimation(AZ::EntityId nodeId, const AZStd::string& animationName);
        void StopNodeAnimation(AZ::EntityId nodeId);
        void SetAnimationSpeed(AZ::EntityId nodeId, float speed);
        
        // Interaction management
        void SetNodeInteractable(AZ::EntityId nodeId, bool interactable);
        void SetNodeHighlighted(AZ::EntityId nodeId, bool highlighted);
        void SetNodeSelected(AZ::EntityId nodeId, bool selected);
        
        // Visual effects
        void EnableNodeParticles(AZ::EntityId nodeId, bool enable);
        void EnableNodeGlow(AZ::EntityId nodeId, bool enable);
        void EnableNodeLighting(AZ::EntityId nodeId, bool enable);
        void SetNodeParticleEffect(AZ::EntityId nodeId, const AZStd::string& effectName);
        
        // Level of detail management
        void SetNodeLOD(AZ::EntityId nodeId, QualityLevel level);
        void UpdateNodeLOD(AZ::EntityId nodeId, float distance);
        void SetNodeVisible(AZ::EntityId nodeId, bool visible);
        
        // Accessibility features
        void SetNodeAccessibilityLabel(AZ::EntityId nodeId, const AZStd::string& label);
        void SetNodeAudioDescription(AZ::EntityId nodeId, const AZStd::string& description);
        void EnableNodeAccessibilityFocus(AZ::EntityId nodeId, bool enable);
        
        // Batch operations
        void UpdateMultipleNodes(const AZStd::vector<AZ::EntityId>& nodeIds, const AZStd::function<void(AZ::EntityId)>& updateFunction);
        void SetNodesVisible(const AZStd::vector<AZ::EntityId>& nodeIds, bool visible);
        void SetNodesQuality(const AZStd::vector<AZ::EntityId>& nodeIds, QualityLevel quality);
        
        // Asset management
        void PreloadNodeAssets(NodeType type);
        void UnloadUnusedAssets();
        void SetAssetQuality(QualityLevel quality);
        
        // Performance optimization
        void EnableInstancing(bool enable);
        void SetMaxVisibleNodes(int maxNodes);
        void EnableFrustumCulling(bool enable);
        void EnableOcclusionCulling(bool enable);
        
        // Utility methods
        NodeVisualizationSpec GetNodeSpec(const WorkflowNode& nodeData) const;
        AZ::Color GetStatusColor(NodeStatusType status) const;
        AZStd::string GetNodeMeshAsset(NodeType type) const;
        AZStd::string GetNodeMaterialAsset(NodeType type) const;
        AZStd::string GetNodeIconAsset(NodeType type) const;
        
        // Statistics
        int GetActiveNodeCount() const;
        int GetVisibleNodeCount() const;
        float GetTotalRenderTime() const;
        int GetTotalDrawCalls() const;
        
        // Debug features
        void EnableDebugMode(bool enable);
        void ShowNodeBounds(AZ::EntityId nodeId, bool show);
        void ShowNodeLabels(bool show);
        void ShowNodeInteractionRadius(bool show);

    private:
        // Core creation helpers
        AZ::EntityId CreateNodeEntity(const NodeVisualizationSpec& spec);
        void SetupNodeTransform(AZ::Entity* entity, const NodeVisualizationSpec& spec);
        void SetupNodeMesh(AZ::Entity* entity, const NodeVisualizationSpec& spec);
        void SetupNodeMaterial(AZ::Entity* entity, const NodeVisualizationSpec& spec);
        void SetupNodeAnimation(AZ::Entity* entity, const NodeVisualizationSpec& spec);
        void SetupNodeInteraction(AZ::Entity* entity, const NodeVisualizationSpec& spec);
        void SetupNodeParticles(AZ::Entity* entity, const NodeVisualizationSpec& spec);
        void SetupNodeLighting(AZ::Entity* entity, const NodeVisualizationSpec& spec);
        void SetupNodeAccessibility(AZ::Entity* entity, const NodeVisualizationSpec& spec);
        void SetupNodeLOD(AZ::Entity* entity, const NodeVisualizationSpec& spec);
        
        // Asset loading
        AZ::Data::Asset<AZ::RPI::ModelAsset> LoadMeshAsset(const AZStd::string& assetPath);
        AZ::Data::Asset<AZ::RPI::MaterialAsset> LoadMaterialAsset(const AZStd::string& assetPath);
        void LoadNodeAssets(NodeType type);
        
        // Material management
        void CreateStatusMaterials();
        void UpdateNodeMaterial(AZ::EntityId nodeId, const AZ::Color& color, float opacity);
        void ApplyStatusMaterial(AZ::EntityId nodeId, NodeStatusType status);
        
        // Animation helpers
        void SetupIdleAnimation(AZ::Entity* entity, NodeType type);
        void SetupExecutionAnimation(AZ::Entity* entity, NodeType type);
        void SetupCompletionAnimation(AZ::Entity* entity, NodeType type);
        void SetupStatusAnimation(AZ::Entity* entity, NodeStatusType status);
        
        // Particle effect helpers
        void SetupDefaultParticleEffects(AZ::Entity* entity, NodeType type);
        void SetupStatusParticleEffects(AZ::Entity* entity, NodeStatusType status);
        void SetupDataFlowParticles(AZ::Entity* entity);
        
        // Lighting helpers
        void SetupNodePointLight(AZ::Entity* entity, const AZ::Color& color, float intensity);
        void SetupNodeSpotLight(AZ::Entity* entity, const AZ::Color& color, float intensity, float angle);
        void SetupNodeDirectionalLight(AZ::Entity* entity, const AZ::Color& color, float intensity);
        
        // Interaction helpers
        void SetupClickInteraction(AZ::Entity* entity, float radius);
        void SetupHoverInteraction(AZ::Entity* entity, float radius);
        void SetupDragInteraction(AZ::Entity* entity);
        void SetupGestureInteraction(AZ::Entity* entity);
        
        // LOD helpers
        void SetupLODLevels(AZ::Entity* entity, QualityLevel baseLevel);
        void UpdateLODBasedOnDistance(AZ::EntityId nodeId, float distance);
        void UpdateLODBasedOnImportance(AZ::EntityId nodeId, float importance);
        
        // Accessibility helpers
        void SetupScreenReaderSupport(AZ::Entity* entity, const AZStd::string& label);
        void SetupAudioDescriptions(AZ::Entity* entity, const AZStd::string& description);
        void SetupKeyboardNavigation(AZ::Entity* entity);
        void SetupVoiceControl(AZ::Entity* entity);
        
        // Performance helpers
        void OptimizeNodeForPerformance(AZ::EntityId nodeId, QualityLevel quality);
        void EnableNodeInstancing(AZ::EntityId nodeId, bool enable);
        void SetupNodeOcclusion(AZ::EntityId nodeId, bool enable);
        void SetupNodeFrustumCulling(AZ::EntityId nodeId, bool enable);
        
        // Cleanup helpers
        void CleanupNodeEntity(AZ::EntityId nodeId);
        void CleanupNodeAssets(AZ::EntityId nodeId);
        void CleanupNodeAnimations(AZ::EntityId nodeId);
        void CleanupNodeParticles(AZ::EntityId nodeId);
        
        // Member variables
        AZStd::unordered_map<AZ::EntityId, NodeVisualizationSpec> m_nodeSpecs;
        AZStd::unordered_map<AZ::EntityId, AZStd::unique_ptr<NodeAnimationController>> m_nodeAnimations;
        AZStd::unordered_map<AZ::EntityId, AZStd::unique_ptr<NodeParticleEffects>> m_nodeParticles;
        AZStd::unordered_map<AZ::EntityId, AZStd::unique_ptr<NodeInteractionHandler>> m_nodeInteractions;
        AZStd::unordered_map<AZ::EntityId, AZStd::unique_ptr<NodeAccessibilityFeatures>> m_nodeAccessibility;
        
        // Asset caches
        AZStd::unordered_map<NodeType, AZ::Data::Asset<AZ::RPI::ModelAsset>> m_meshAssets;
        AZStd::unordered_map<NodeType, AZ::Data::Asset<AZ::RPI::MaterialAsset>> m_materialAssets;
        AZStd::unordered_map<NodeStatusType, AZ::Data::Asset<AZ::RPI::MaterialAsset>> m_statusMaterials;
        
        // Configuration
        QualityLevel m_currentQuality = QualityLevel::Medium;
        bool m_instancingEnabled = true;
        int m_maxVisibleNodes = 1000;
        bool m_frustumCullingEnabled = true;
        bool m_occlusionCullingEnabled = true;
        bool m_debugMode = false;
        
        // Performance tracking
        int m_activeNodeCount = 0;
        int m_visibleNodeCount = 0;
        float m_totalRenderTime = 0.0f;
        int m_totalDrawCalls = 0;
        
        // Asset paths
        static const AZStd::unordered_map<NodeType, AZStd::string> s_meshAssetPaths;
        static const AZStd::unordered_map<NodeType, AZStd::string> s_materialAssetPaths;
        static const AZStd::unordered_map<NodeType, AZStd::string> s_iconAssetPaths;
        static const AZStd::unordered_map<NodeStatusType, AZ::Color> s_statusColors;
    };

} // namespace Q3D 