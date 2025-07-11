#include "WorkflowVisualization3DSystemComponent.h"
#include <AzCore/Serialization/SerializeContext.h>
#include <AzCore/Serialization/EditContext.h>
#include <AzCore/Console/ILogger.h>
#include <AzCore/Component/ComponentApplicationBus.h>
#include <AzCore/Component/Entity.h>
#include <AzCore/Math/Transform.h>
#include <AzCore/Math/Vector3.h>
#include <AzCore/Math/Quaternion.h>
#include <AzCore/std/string/string.h>
#include <AzCore/std/containers/vector.h>
#include <AzCore/std/smart_ptr/make_shared.h>
#include <AzFramework/Entity/EntityContextBus.h>
#include <AzFramework/Entity/GameEntityContextBus.h>
#include <AzFramework/Render/GeometryIntersectionBus.h>
#include <O3DE/Entity/Entity.h>
#include <O3DE/Transform/TransformBus.h>
#include <O3DE/Render/MeshComponentBus.h>
#include <O3DE/Render/MaterialAsset.h>

#ifdef EMSCRIPTEN
#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#endif

namespace Q3D
{
    void WorkflowVisualization3DSystemComponent::Reflect(AZ::ReflectContext* context)
    {
        if (AZ::SerializeContext* serialize = azrtti_cast<AZ::SerializeContext*>(context))
        {
            serialize->Class<WorkflowVisualization3DSystemComponent, AZ::Component>()
                ->Version(0)
                ;

            if (AZ::EditContext* ec = serialize->GetEditContext())
            {
                ec->Class<WorkflowVisualization3DSystemComponent>("WorkflowVisualization3D", "3D Workflow Visualization System")
                    ->ClassElement(AZ::Edit::ClassElements::EditorData, "")
                    ->Attribute(AZ::Edit::Attributes::AppearsInAddComponentMenu, AZ_CRC("System"))
                    ->Attribute(AZ::Edit::Attributes::AutoExpand, true)
                    ;
            }
        }
    }

    void WorkflowVisualization3DSystemComponent::GetProvidedServices(AZ::ComponentDescriptor::DependencyArrayType& provided)
    {
        provided.push_back(AZ_CRC("WorkflowVisualization3DService"));
    }

    void WorkflowVisualization3DSystemComponent::GetIncompatibleServices(AZ::ComponentDescriptor::DependencyArrayType& incompatible)
    {
        incompatible.push_back(AZ_CRC("WorkflowVisualization3DService"));
    }

    void WorkflowVisualization3DSystemComponent::GetRequiredServices(AZ::ComponentDescriptor::DependencyArrayType& required)
    {
        AZ_UNUSED(required);
    }

    void WorkflowVisualization3DSystemComponent::GetDependentServices(AZ::ComponentDescriptor::DependencyArrayType& dependent)
    {
        AZ_UNUSED(dependent);
    }

    WorkflowVisualization3DSystemComponent::WorkflowVisualization3DSystemComponent()
        : m_currentMode(VisualizationMode::ThreeD)
        , m_currentQuality(QualityLevel::Medium)
        , m_collaborationEnabled(false)
        , m_timeSinceLastUpdate(0.0f)
        , m_targetUpdateRate(1.0f / 60.0f)
        , m_needsRender(false)
        , m_isInitialized(false)
    {
        AZ_Printf("WorkflowVisualization3D", "WorkflowVisualization3DSystemComponent created");
    }

    WorkflowVisualization3DSystemComponent::~WorkflowVisualization3DSystemComponent()
    {
        AZ_Printf("WorkflowVisualization3D", "WorkflowVisualization3DSystemComponent destroyed");
    }

    void WorkflowVisualization3DSystemComponent::Init()
    {
        AZ_Printf("WorkflowVisualization3D", "Initializing WorkflowVisualization3D System Component");
        
        // Initialize core components
        InitializeComponents();
        
        // Setup WebAssembly bindings if building for web
        #ifdef EMSCRIPTEN
        SetupWebAssemblyBindings();
        #endif
        
        AZ_Printf("WorkflowVisualization3D", "WorkflowVisualization3D System Component initialized");
    }

    void WorkflowVisualization3DSystemComponent::Activate()
    {
        AZ_Printf("WorkflowVisualization3D", "Activating WorkflowVisualization3D System Component");
        
        // Connect to buses
        WorkflowVisualization3DRequestBus::Handler::BusConnect();
        AZ::TickBus::Handler::BusConnect();
        
        // Initialize all subsystems
        if (m_nodeFactory)
        {
            m_nodeFactory->Initialize();
        }
        
        if (m_collaborationManager)
        {
            m_collaborationManager->Initialize();
        }
        
        if (m_performanceMonitor)
        {
            m_performanceMonitor->Initialize();
        }
        
        if (m_spatialAudioManager)
        {
            m_spatialAudioManager->Initialize();
        }
        
        if (m_aiLayoutOptimizer)
        {
            m_aiLayoutOptimizer->Initialize();
        }
        
        if (m_accessibilityManager)
        {
            m_accessibilityManager->Initialize();
        }
        
        // Set initialization flag
        m_isInitialized = true;
        
        AZ_Printf("WorkflowVisualization3D", "WorkflowVisualization3D System Component activated");
    }

    void WorkflowVisualization3DSystemComponent::Deactivate()
    {
        AZ_Printf("WorkflowVisualization3D", "Deactivating WorkflowVisualization3D System Component");
        
        // Set initialization flag
        m_isInitialized = false;
        
        // Shutdown all subsystems
        if (m_accessibilityManager)
        {
            m_accessibilityManager->Shutdown();
        }
        
        if (m_aiLayoutOptimizer)
        {
            m_aiLayoutOptimizer->Shutdown();
        }
        
        if (m_spatialAudioManager)
        {
            m_spatialAudioManager->Shutdown();
        }
        
        if (m_performanceMonitor)
        {
            m_performanceMonitor->Shutdown();
        }
        
        if (m_collaborationManager)
        {
            m_collaborationManager->Shutdown();
        }
        
        if (m_nodeFactory)
        {
            m_nodeFactory->Shutdown();
        }
        
        // Cleanup workflow data
        CleanupWorkflow();
        
        // Disconnect from buses
        AZ::TickBus::Handler::BusDisconnect();
        WorkflowVisualization3DRequestBus::Handler::BusDisconnect();
        
        AZ_Printf("WorkflowVisualization3D", "WorkflowVisualization3D System Component deactivated");
    }

    void WorkflowVisualization3DSystemComponent::OnTick(float deltaTime, AZ::ScriptTimePoint time)
    {
        AZ_UNUSED(time);
        
        if (!m_isInitialized)
        {
            return;
        }
        
        // Update timing
        m_timeSinceLastUpdate += deltaTime;
        
        // Check if we need to update
        if (m_timeSinceLastUpdate >= m_targetUpdateRate)
        {
            UpdateVisualization(deltaTime);
            m_timeSinceLastUpdate = 0.0f;
        }
        
        // Update subsystems
        UpdateCollaborativeUsers(deltaTime);
        ProcessPendingInteractions();
        
        // Update performance monitoring
        if (m_performanceMonitor)
        {
            m_performanceMonitor->Update(deltaTime);
        }
        
        // Update spatial audio
        if (m_spatialAudioManager)
        {
            m_spatialAudioManager->Update(deltaTime);
        }
        
        // Update accessibility features
        if (m_accessibilityManager)
        {
            m_accessibilityManager->Update(deltaTime);
        }
    }

    void WorkflowVisualization3DSystemComponent::CreateWorkflowVisualization(const WorkflowData& workflowData)
    {
        AZ_Printf("WorkflowVisualization3D", "Creating workflow visualization for: %s", workflowData.workflowId.c_str());
        
        if (!m_isInitialized)
        {
            AZ_Error("WorkflowVisualization3D", false, "System not initialized");
            return;
        }
        
        // Clean up existing workflow
        CleanupWorkflow();
        
        // Store workflow data
        m_currentWorkflow = workflowData;
        
        // Create 3D nodes for each workflow node
        for (const auto& node : workflowData.nodes)
        {
            if (m_nodeFactory)
            {
                AZ::EntityId nodeEntity = m_nodeFactory->CreateWorkflowNode(node);
                if (nodeEntity.IsValid())
                {
                    m_nodeEntities[node.nodeId] = nodeEntity;
                }
            }
        }
        
        // Create 3D connections between nodes
        for (const auto& connection : workflowData.connections)
        {
            if (m_nodeFactory)
            {
                AZ::EntityId edgeEntity = m_nodeFactory->CreateWorkflowConnection(connection);
                if (edgeEntity.IsValid())
                {
                    AZStd::string edgeId = AZStd::string::format("%s_%s", 
                        connection.fromNodeId.c_str(), connection.toNodeId.c_str());
                    m_edgeEntities[edgeId] = edgeEntity;
                }
            }
        }
        
        // Apply initial layout optimization
        if (m_aiLayoutOptimizer)
        {
            LayoutOptimizationParams params;
            params.workflow = workflowData;
            params.canvasSize = AZ::Vector3(20.0f, 20.0f, 20.0f);
            params.layoutAlgorithm = "ai_optimized";
            
            AZStd::string optimizationId = m_aiLayoutOptimizer->OptimizeLayout(params);
            AZ_Printf("WorkflowVisualization3D", "Layout optimization started: %s", optimizationId.c_str());
        }
        
        // Mark as needing render
        m_needsRender = true;
        
        // Notify listeners
        OnWorkflowDataChanged();
        
        AZ_Printf("WorkflowVisualization3D", "Workflow visualization created successfully");
    }

    void WorkflowVisualization3DSystemComponent::UpdateWorkflowVisualization(const WorkflowData& workflowData)
    {
        AZ_Printf("WorkflowVisualization3D", "Updating workflow visualization for: %s", workflowData.workflowId.c_str());
        
        if (!m_isInitialized)
        {
            AZ_Error("WorkflowVisualization3D", false, "System not initialized");
            return;
        }
        
        // Update workflow data
        m_currentWorkflow = workflowData;
        
        // Update existing nodes
        for (const auto& node : workflowData.nodes)
        {
            auto nodeIt = m_nodeEntities.find(node.nodeId);
            if (nodeIt != m_nodeEntities.end())
            {
                // Update existing node
                if (m_nodeFactory)
                {
                    m_nodeFactory->UpdateWorkflowNode(nodeIt->second, node);
                }
            }
            else
            {
                // Create new node
                if (m_nodeFactory)
                {
                    AZ::EntityId nodeEntity = m_nodeFactory->CreateWorkflowNode(node);
                    if (nodeEntity.IsValid())
                    {
                        m_nodeEntities[node.nodeId] = nodeEntity;
                    }
                }
            }
        }
        
        // Update connections
        for (const auto& connection : workflowData.connections)
        {
            AZStd::string edgeId = AZStd::string::format("%s_%s", 
                connection.fromNodeId.c_str(), connection.toNodeId.c_str());
            
            auto edgeIt = m_edgeEntities.find(edgeId);
            if (edgeIt != m_edgeEntities.end())
            {
                // Update existing connection
                if (m_nodeFactory)
                {
                    m_nodeFactory->UpdateWorkflowConnection(edgeIt->second, connection);
                }
            }
            else
            {
                // Create new connection
                if (m_nodeFactory)
                {
                    AZ::EntityId edgeEntity = m_nodeFactory->CreateWorkflowConnection(connection);
                    if (edgeEntity.IsValid())
                    {
                        m_edgeEntities[edgeId] = edgeEntity;
                    }
                }
            }
        }
        
        // Mark as needing render
        m_needsRender = true;
        
        // Notify listeners
        OnWorkflowDataChanged();
        
        AZ_Printf("WorkflowVisualization3D", "Workflow visualization updated successfully");
    }

    void WorkflowVisualization3DSystemComponent::UpdateNodeStatus(const AZStd::string& nodeId, const NodeStatus& status)
    {
        AZ_Printf("WorkflowVisualization3D", "Updating node status for: %s", nodeId.c_str());
        
        if (!m_isInitialized)
        {
            AZ_Error("WorkflowVisualization3D", false, "System not initialized");
            return;
        }
        
        auto nodeIt = m_nodeEntities.find(nodeId);
        if (nodeIt != m_nodeEntities.end())
        {
            if (m_nodeFactory)
            {
                m_nodeFactory->UpdateNodeStatus(nodeIt->second, status);
            }
            
            // Update accessibility announcements
            if (m_accessibilityManager)
            {
                AZStd::string statusMessage = AZStd::string::format("Node %s is now %s", 
                    nodeId.c_str(), NodeStatusTypeToString(status.status).c_str());
                m_accessibilityManager->AnnounceText(statusMessage);
            }
            
            // Notify listeners
            OnNodeStatusChanged(nodeId, status);
        }
        else
        {
            AZ_Warning("WorkflowVisualization3D", false, "Node not found: %s", nodeId.c_str());
        }
    }

    void WorkflowVisualization3DSystemComponent::SetVisualizationMode(VisualizationMode mode)
    {
        AZ_Printf("WorkflowVisualization3D", "Setting visualization mode to: %d", static_cast<int>(mode));
        
        if (m_currentMode != mode)
        {
            m_currentMode = mode;
            
            // Update rendering mode
            if (m_nodeFactory)
            {
                m_nodeFactory->SetVisualizationMode(mode);
            }
            
            // Update accessibility features based on mode
            if (m_accessibilityManager)
            {
                switch (mode)
                {
                case VisualizationMode::VR:
                    m_accessibilityManager->EnableAccessibilityFeature(AccessibilityFeature::SpatialAudio, true);
                    m_accessibilityManager->EnableAccessibilityFeature(AccessibilityFeature::HapticFeedback, true);
                    break;
                case VisualizationMode::AR:
                    m_accessibilityManager->EnableAccessibilityFeature(AccessibilityFeature::GestureAlternatives, true);
                    break;
                default:
                    break;
                }
            }
            
            // Mark as needing render
            m_needsRender = true;
            
            // Notify listeners
            WorkflowVisualization3DNotificationBus::Broadcast(&WorkflowVisualization3DNotifications::OnVisualizationModeChanged, mode);
        }
    }

    void WorkflowVisualization3DSystemComponent::SetQualityLevel(QualityLevel level)
    {
        AZ_Printf("WorkflowVisualization3D", "Setting quality level to: %d", static_cast<int>(level));
        
        if (m_currentQuality != level)
        {
            m_currentQuality = level;
            
            // Update performance settings
            if (m_performanceMonitor)
            {
                m_performanceMonitor->ApplyQualitySettings(level);
            }
            
            // Update node factory quality settings
            if (m_nodeFactory)
            {
                m_nodeFactory->SetQualityLevel(level);
            }
            
            // Mark as needing render
            m_needsRender = true;
            
            // Notify listeners
            WorkflowVisualization3DNotificationBus::Broadcast(&WorkflowVisualization3DNotifications::OnQualityLevelChanged, level);
        }
    }

    void WorkflowVisualization3DSystemComponent::EnableCollaboration(bool enable, const AZStd::string& sessionId)
    {
        AZ_Printf("WorkflowVisualization3D", "Setting collaboration enabled: %s, session: %s", 
            enable ? "true" : "false", sessionId.c_str());
        
        if (m_collaborationEnabled != enable)
        {
            m_collaborationEnabled = enable;
            m_sessionId = sessionId;
            
            if (m_collaborationManager)
            {
                if (enable)
                {
                    m_collaborationManager->StartSession(sessionId);
                    
                    // Enable spatial audio for collaboration
                    if (m_spatialAudioManager)
                    {
                        m_spatialAudioManager->EnableSpatialAudio(true);
                    }
                    
                    // Notify listeners
                    WorkflowVisualization3DNotificationBus::Broadcast(&WorkflowVisualization3DNotifications::OnCollaborationEnabled, sessionId);
                }
                else
                {
                    m_collaborationManager->EndSession();
                    
                    // Disable spatial audio
                    if (m_spatialAudioManager)
                    {
                        m_spatialAudioManager->EnableSpatialAudio(false);
                    }
                    
                    // Clear users
                    m_collaborativeUsers.clear();
                    
                    // Notify listeners
                    WorkflowVisualization3DNotificationBus::Broadcast(&WorkflowVisualization3DNotifications::OnCollaborationDisabled);
                }
            }
        }
    }

    void WorkflowVisualization3DSystemComponent::AddCollaborativeUser(const UserPresence& user)
    {
        AZ_Printf("WorkflowVisualization3D", "Adding collaborative user: %s", user.userId.c_str());
        
        if (!m_collaborationEnabled)
        {
            AZ_Warning("WorkflowVisualization3D", false, "Collaboration not enabled");
            return;
        }
        
        // Store user data
        m_collaborativeUsers[user.userId] = user;
        
        // Add user to collaboration manager
        if (m_collaborationManager)
        {
            m_collaborationManager->AddUser(user);
        }
        
        // Add user to spatial audio
        if (m_spatialAudioManager)
        {
            m_spatialAudioManager->AddUser(user.userId, user.position);
        }
        
        // Announce user joining for accessibility
        if (m_accessibilityManager)
        {
            AZStd::string announcement = AZStd::string::format("%s joined the collaboration session", 
                user.displayName.c_str());
            m_accessibilityManager->AnnounceText(announcement);
        }
        
        // Notify listeners
        WorkflowVisualization3DNotificationBus::Broadcast(&WorkflowVisualization3DNotifications::OnUserJoined, user);
    }

    void WorkflowVisualization3DSystemComponent::RemoveCollaborativeUser(const AZStd::string& userId)
    {
        AZ_Printf("WorkflowVisualization3D", "Removing collaborative user: %s", userId.c_str());
        
        auto userIt = m_collaborativeUsers.find(userId);
        if (userIt != m_collaborativeUsers.end())
        {
            AZStd::string displayName = userIt->second.displayName;
            
            // Remove from collaboration manager
            if (m_collaborationManager)
            {
                m_collaborationManager->RemoveUser(userId);
            }
            
            // Remove from spatial audio
            if (m_spatialAudioManager)
            {
                m_spatialAudioManager->RemoveUser(userId);
            }
            
            // Remove from our list
            m_collaborativeUsers.erase(userIt);
            
            // Announce user leaving for accessibility
            if (m_accessibilityManager)
            {
                AZStd::string announcement = AZStd::string::format("%s left the collaboration session", 
                    displayName.c_str());
                m_accessibilityManager->AnnounceText(announcement);
            }
            
            // Notify listeners
            WorkflowVisualization3DNotificationBus::Broadcast(&WorkflowVisualization3DNotifications::OnUserLeft, userId);
        }
    }

    void WorkflowVisualization3DSystemComponent::UpdateUserCursor(const AZStd::string& userId, const AZ::Vector3& position)
    {
        auto userIt = m_collaborativeUsers.find(userId);
        if (userIt != m_collaborativeUsers.end())
        {
            userIt->second.position = position;
            
            // Update in collaboration manager
            if (m_collaborationManager)
            {
                m_collaborationManager->UpdateUserCursor(userId, position);
            }
            
            // Update spatial audio position
            if (m_spatialAudioManager)
            {
                m_spatialAudioManager->UpdateUserPosition(userId, position);
            }
        }
    }

    void WorkflowVisualization3DSystemComponent::HandleUserInteraction(const UserInteraction& interaction)
    {
        AZ_Printf("WorkflowVisualization3D", "Handling user interaction: %s", interaction.interactionType.c_str());
        
        if (m_collaborationManager)
        {
            m_collaborationManager->HandleUserInteraction(interaction);
        }
        
        // Handle accessibility interactions
        if (m_accessibilityManager)
        {
            if (interaction.interactionType == "voice_command")
            {
                auto paramIt = interaction.parameters.find("command");
                if (paramIt != interaction.parameters.end())
                {
                    m_accessibilityManager->ProcessVoiceCommand(paramIt->second);
                }
            }
            else if (interaction.interactionType == "keyboard_navigation")
            {
                auto paramIt = interaction.parameters.find("key");
                if (paramIt != interaction.parameters.end())
                {
                    m_accessibilityManager->HandleKeyboardInput(paramIt->second);
                }
            }
        }
        
        // Notify listeners
        WorkflowVisualization3DNotificationBus::Broadcast(&WorkflowVisualization3DNotifications::OnNodeInteraction, 
            interaction.targetNodeId, interaction);
    }

    AZStd::string WorkflowVisualization3DSystemComponent::OptimizeLayout(const LayoutOptimizationParams& params)
    {
        AZ_Printf("WorkflowVisualization3D", "Optimizing layout with algorithm: %s", params.layoutAlgorithm.c_str());
        
        if (!m_isInitialized || !m_aiLayoutOptimizer)
        {
            AZ_Error("WorkflowVisualization3D", false, "System not initialized or AI optimizer not available");
            return "";
        }
        
        return m_aiLayoutOptimizer->OptimizeLayout(params);
    }

    PerformanceMetrics WorkflowVisualization3DSystemComponent::GetPerformanceMetrics() const
    {
        if (m_performanceMonitor)
        {
            return m_performanceMonitor->GetCurrentMetrics();
        }
        
        return PerformanceMetrics();
    }

    bool WorkflowVisualization3DSystemComponent::IsInitialized() const
    {
        return m_isInitialized;
    }

    // Accessibility Methods
    void WorkflowVisualization3DSystemComponent::EnableAccessibilityFeature(AccessibilityFeature feature, bool enabled)
    {
        if (m_accessibilityManager)
        {
            m_accessibilityManager->SetFeatureEnabled(feature, enabled);
        }
    }

    void WorkflowVisualization3DSystemComponent::SetAccessibilityProfile(const AZStd::string& profileId)
    {
        if (m_accessibilityManager)
        {
            m_accessibilityManager->LoadAccessibilityProfile(profileId);
        }
    }

    void WorkflowVisualization3DSystemComponent::SetAccessibilityLevel(AccessibilityLevel level)
    {
        if (m_accessibilityManager)
        {
            m_accessibilityManager->SetAccessibilityLevel(level);
        }
    }

    void WorkflowVisualization3DSystemComponent::AnnounceToScreenReader(const AZStd::string& message)
    {
        if (m_accessibilityManager)
        {
            m_accessibilityManager->AnnounceText(message);
        }
    }

    void WorkflowVisualization3DSystemComponent::TriggerHapticFeedback(const AZStd::string& feedbackType, float intensity)
    {
        if (m_accessibilityManager)
        {
            m_accessibilityManager->TriggerHapticFeedback(feedbackType, intensity);
        }
    }

    void WorkflowVisualization3DSystemComponent::UpdateFocusedElement(const AZStd::string& elementId)
    {
        if (m_accessibilityManager)
        {
            m_accessibilityManager->SetFocusedElement(elementId);
        }
    }

    bool WorkflowVisualization3DSystemComponent::ProcessVoiceCommand(const AZStd::string& command)
    {
        if (m_accessibilityManager)
        {
            return m_accessibilityManager->ProcessVoiceCommand(command);
        }
        return false;
    }

    void WorkflowVisualization3DSystemComponent::ValidateAccessibilityCompliance()
    {
        if (m_accessibilityManager)
        {
            m_accessibilityManager->ValidateAccessibility();
        }
    }

    AccessibilitySettings WorkflowVisualization3DSystemComponent::GetAccessibilitySettings() const
    {
        if (m_accessibilityManager)
        {
            return m_accessibilityManager->GetAccessibilitySettings();
        }
        return AccessibilitySettings();
    }

    // Private Methods
    void WorkflowVisualization3DSystemComponent::InitializeComponents()
    {
        AZ_Printf("WorkflowVisualization3D", "Initializing core components");
        
        // Initialize node factory
        m_nodeFactory = AZStd::make_unique<NodeFactory>();
        
        // Initialize collaboration manager
        m_collaborationManager = AZStd::make_unique<CollaborationManager>();
        
        // Initialize performance monitor
        m_performanceMonitor = AZStd::make_unique<PerformanceMonitor>();
        
        // Initialize spatial audio manager
        m_spatialAudioManager = AZStd::make_unique<SpatialAudioManager>();
        
        // Initialize AI layout optimizer
        m_aiLayoutOptimizer = AZStd::make_unique<AILayoutOptimizer>();
        
        // Initialize accessibility manager
        m_accessibilityManager = AZStd::make_unique<AccessibilityManager>();
        
        // Initialize interaction system
        m_interactionSystem = AZStd::make_unique<InteractionSystem>();
        
        AZ_Printf("WorkflowVisualization3D", "Core components initialized");
    }

    void WorkflowVisualization3DSystemComponent::UpdateVisualization(float deltaTime)
    {
        if (!m_needsRender)
        {
            return;
        }
        
        // Update node animations
        if (m_nodeFactory)
        {
            m_nodeFactory->UpdateAnimations(deltaTime);
        }
        
        // Update performance optimization
        OptimizeRenderingPerformance();
        
        // Update level of detail
        UpdateLevelOfDetail();
        
        // Cull invisible nodes
        CullInvisibleNodes();
        
        // Clear render flag
        m_needsRender = false;
    }

    void WorkflowVisualization3DSystemComponent::CleanupWorkflow()
    {
        AZ_Printf("WorkflowVisualization3D", "Cleaning up workflow");
        
        // Destroy node entities
        for (auto& pair : m_nodeEntities)
        {
            if (pair.second.IsValid())
            {
                AZ::ComponentApplicationBus::Broadcast(&AZ::ComponentApplicationBus::Events::DeleteEntity, pair.second);
            }
        }
        m_nodeEntities.clear();
        
        // Destroy edge entities
        for (auto& pair : m_edgeEntities)
        {
            if (pair.second.IsValid())
            {
                AZ::ComponentApplicationBus::Broadcast(&AZ::ComponentApplicationBus::Events::DeleteEntity, pair.second);
            }
        }
        m_edgeEntities.clear();
        
        // Clear workflow data
        m_currentWorkflow = WorkflowData();
    }

    void WorkflowVisualization3DSystemComponent::UpdateCollaborativeUsers(float deltaTime)
    {
        if (!m_collaborationEnabled)
        {
            return;
        }
        
        // Update user activity timers
        for (auto& pair : m_collaborativeUsers)
        {
            pair.second.lastActivity += deltaTime;
        }
        
        // Update collaboration manager
        if (m_collaborationManager)
        {
            m_collaborationManager->Update(deltaTime);
        }
    }

    void WorkflowVisualization3DSystemComponent::ProcessPendingInteractions()
    {
        if (m_interactionSystem)
        {
            m_interactionSystem->ProcessPendingInteractions();
        }
    }

    void WorkflowVisualization3DSystemComponent::OptimizeRenderingPerformance()
    {
        if (m_performanceMonitor)
        {
            PerformanceMetrics metrics = m_performanceMonitor->GetCurrentMetrics();
            
            // Auto-adjust quality based on performance
            if (metrics.frameTime > 33.0f) // Below 30 FPS
            {
                QualityLevel newLevel = static_cast<QualityLevel>(static_cast<int>(m_currentQuality) - 1);
                if (newLevel >= QualityLevel::Low)
                {
                    SetQualityLevel(newLevel);
                }
            }
            else if (metrics.frameTime < 13.0f) // Above 75 FPS
            {
                QualityLevel newLevel = static_cast<QualityLevel>(static_cast<int>(m_currentQuality) + 1);
                if (newLevel <= QualityLevel::Ultra)
                {
                    SetQualityLevel(newLevel);
                }
            }
        }
    }

    void WorkflowVisualization3DSystemComponent::UpdateLevelOfDetail()
    {
        if (m_nodeFactory)
        {
            m_nodeFactory->UpdateLevelOfDetail();
        }
    }

    void WorkflowVisualization3DSystemComponent::CullInvisibleNodes()
    {
        if (m_nodeFactory)
        {
            m_nodeFactory->CullInvisibleNodes();
        }
    }

    void WorkflowVisualization3DSystemComponent::OnWorkflowDataChanged()
    {
        WorkflowVisualization3DNotificationBus::Broadcast(&WorkflowVisualization3DNotifications::OnWorkflowVisualizationUpdated, 
            m_currentWorkflow.workflowId);
    }

    void WorkflowVisualization3DSystemComponent::OnNodeStatusChanged(const AZStd::string& nodeId, const NodeStatus& status)
    {
        WorkflowVisualization3DNotificationBus::Broadcast(&WorkflowVisualization3DNotifications::OnNodeStatusChanged, 
            nodeId, status);
    }

    AZStd::string WorkflowVisualization3DSystemComponent::NodeStatusTypeToString(NodeStatusType status)
    {
        switch (status)
        {
        case NodeStatusType::Pending: return "pending";
        case NodeStatusType::Running: return "running";
        case NodeStatusType::Completed: return "completed";
        case NodeStatusType::Failed: return "failed";
        case NodeStatusType::PendingApproval: return "pending approval";
        case NodeStatusType::Cancelled: return "cancelled";
        default: return "unknown";
        }
    }

    #ifdef EMSCRIPTEN
    void WorkflowVisualization3DSystemComponent::SetupWebAssemblyBindings()
    {
        AZ_Printf("WorkflowVisualization3D", "Setting up WebAssembly bindings");
        
        // WebAssembly bindings will be set up through the C interface functions
        // defined in the header file
    }
    #endif

    // WebAssembly C interface functions
    #ifdef EMSCRIPTEN
    extern "C" {
        EMSCRIPTEN_KEEPALIVE void CreateWorkflowVisualization(const char* workflowDataJson)
        {
            // Implementation would parse JSON and call the appropriate method
            AZ_Printf("WorkflowVisualization3D", "WebAssembly: CreateWorkflowVisualization called");
        }
        
        EMSCRIPTEN_KEEPALIVE void UpdateNodeStatus(const char* nodeId, const char* statusJson)
        {
            // Implementation would parse JSON and call the appropriate method
            AZ_Printf("WorkflowVisualization3D", "WebAssembly: UpdateNodeStatus called for %s", nodeId);
        }
        
        EMSCRIPTEN_KEEPALIVE void HandleUserInteraction(const char* interactionJson)
        {
            // Implementation would parse JSON and call the appropriate method
            AZ_Printf("WorkflowVisualization3D", "WebAssembly: HandleUserInteraction called");
        }
        
        EMSCRIPTEN_KEEPALIVE void SetCollaborationMode(bool enabled, const char* sessionId)
        {
            WorkflowVisualization3DRequestBus::Broadcast(&WorkflowVisualization3DRequests::EnableCollaboration, 
                enabled, sessionId);
        }
        
        EMSCRIPTEN_KEEPALIVE void AddCollaborativeUser(const char* userPresenceJson)
        {
            // Implementation would parse JSON and call the appropriate method
            AZ_Printf("WorkflowVisualization3D", "WebAssembly: AddCollaborativeUser called");
        }
        
        EMSCRIPTEN_KEEPALIVE void RemoveCollaborativeUser(const char* userId)
        {
            WorkflowVisualization3DRequestBus::Broadcast(&WorkflowVisualization3DRequests::RemoveCollaborativeUser, 
                userId);
        }
        
        EMSCRIPTEN_KEEPALIVE void UpdateUserCursor(const char* userId, float x, float y, float z)
        {
            AZ::Vector3 position(x, y, z);
            WorkflowVisualization3DRequestBus::Broadcast(&WorkflowVisualization3DRequests::UpdateUserCursor, 
                userId, position);
        }
        
        EMSCRIPTEN_KEEPALIVE void SetVisualizationMode(int mode)
        {
            WorkflowVisualization3DRequestBus::Broadcast(&WorkflowVisualization3DRequests::SetVisualizationMode, 
                static_cast<VisualizationMode>(mode));
        }
        
        EMSCRIPTEN_KEEPALIVE void SetQualityLevel(int level)
        {
            WorkflowVisualization3DRequestBus::Broadcast(&WorkflowVisualization3DRequests::SetQualityLevel, 
                static_cast<QualityLevel>(level));
        }
        
        EMSCRIPTEN_KEEPALIVE const char* OptimizeLayout(const char* paramsJson)
        {
            // Implementation would parse JSON and call the appropriate method
            AZ_Printf("WorkflowVisualization3D", "WebAssembly: OptimizeLayout called");
            return "optimization_id_placeholder";
        }
        
        EMSCRIPTEN_KEEPALIVE const char* GetPerformanceMetrics()
        {
            // Implementation would return JSON performance metrics
            AZ_Printf("WorkflowVisualization3D", "WebAssembly: GetPerformanceMetrics called");
            return "{}";
        }
        
        EMSCRIPTEN_KEEPALIVE bool IsSystemInitialized()
        {
            bool initialized = false;
            WorkflowVisualization3DRequestBus::BroadcastResult(initialized, 
                &WorkflowVisualization3DRequests::IsInitialized);
            return initialized;
        }
    }
    #endif

} // namespace Q3D 