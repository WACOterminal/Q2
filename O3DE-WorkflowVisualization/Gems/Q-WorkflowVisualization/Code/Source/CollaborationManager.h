#pragma once

#include <AzCore/std/containers/unordered_map.h>
#include <AzCore/std/containers/vector.h>
#include <AzCore/std/string/string.h>
#include <AzCore/std/smart_ptr/unique_ptr.h>
#include <AzCore/std/functional.h>
#include <AzCore/Math/Vector3.h>
#include <AzCore/Math/Color.h>
#include <AzCore/Component/TickBus.h>
#include <AzCore/std/chrono/chrono.h>
#include <AzCore/std/parallel/mutex.h>
#include <AzCore/std/parallel/thread.h>

#include "WorkflowVisualization3DBus.h"
#include "SpatialAudioManager.h"
#include "NetworkManager.h"

namespace Q3D
{
    // Forward declarations
    class NetworkManager;
    class SpatialAudioManager;
    class ConflictResolutionManager;
    class CollaborationSecurityManager;

    enum class CollaborationEventType : int
    {
        UserJoined = 0,
        UserLeft = 1,
        UserCursorMove = 2,
        UserNodeEdit = 3,
        UserCommentAdd = 4,
        UserVoiceData = 5,
        UserGesture = 6,
        WorkflowLock = 7,
        WorkflowUnlock = 8,
        ConflictDetected = 9,
        ConflictResolved = 10,
        SessionCreated = 11,
        SessionDestroyed = 12,
        PermissionChanged = 13,
        ScreenShare = 14,
        FileTransfer = 15
    };

    enum class UserPermission : int
    {
        Viewer = 0,
        Editor = 1,
        Moderator = 2,
        Admin = 3
    };

    struct CollaborationSession
    {
        AZStd::string sessionId;
        AZStd::string name;
        AZStd::string description;
        AZStd::string workflowId;
        AZStd::string ownerId;
        AZStd::string createdAt;
        AZStd::string updatedAt;
        bool isActive = true;
        bool isPublic = false;
        bool voiceEnabled = true;
        bool screenShareEnabled = true;
        bool fileTransferEnabled = true;
        int maxUsers = 10;
        AZStd::string password;
        AZStd::unordered_map<AZStd::string, AZStd::string> metadata;
        
        AZ_TYPE_INFO(CollaborationSession, "{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE}");
    };

    struct UserCursor
    {
        AZ::Vector3 position;
        AZ::Vector3 direction;
        AZ::Color color;
        float size = 1.0f;
        bool visible = true;
        AZStd::string currentAction;
        AZStd::string targetNodeId;
        float lastUpdateTime = 0.0f;
        AZStd::unordered_map<AZStd::string, AZStd::string> metadata;
        
        AZ_TYPE_INFO(UserCursor, "{BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF}");
    };

    struct UserEditLock
    {
        AZStd::string userId;
        AZStd::string nodeId;
        AZStd::string lockType; // "exclusive", "shared", "soft"
        float lockTime = 0.0f;
        float expirationTime = 0.0f;
        AZStd::string lockReason;
        bool isActive = true;
        
        AZ_TYPE_INFO(UserEditLock, "{CCCCCCCC-DDDD-EEEE-FFFF-000000000000}");
    };

    struct CollaborationComment
    {
        AZStd::string commentId;
        AZStd::string userId;
        AZStd::string userName;
        AZStd::string message;
        AZ::Vector3 position;
        AZStd::string targetNodeId;
        AZStd::string createdAt;
        AZStd::string updatedAt;
        bool isResolved = false;
        AZStd::vector<AZStd::string> replies;
        AZStd::string threadId;
        
        AZ_TYPE_INFO(CollaborationComment, "{DDDDDDDD-EEEE-FFFF-0000-111111111111}");
    };

    struct UserVoiceData
    {
        AZStd::string userId;
        AZStd::vector<AZ::u8> audioData;
        float timestamp = 0.0f;
        int sampleRate = 44100;
        int channels = 1;
        AZStd::string codec;
        bool isSpatial = true;
        AZ::Vector3 position;
        
        AZ_TYPE_INFO(UserVoiceData, "{EEEEEEEE-FFFF-0000-1111-222222222222}");
    };

    struct CollaborationConflict
    {
        AZStd::string conflictId;
        AZStd::string nodeId;
        AZStd::vector<AZStd::string> involvedUsers;
        AZStd::string conflictType; // "simultaneous_edit", "version_mismatch", "permission_denied"
        AZStd::string description;
        AZStd::string detectedAt;
        AZStd::string resolvedAt;
        bool isResolved = false;
        AZStd::string resolutionStrategy;
        AZStd::unordered_map<AZStd::string, AZStd::string> conflictData;
        
        AZ_TYPE_INFO(CollaborationConflict, "{FFFFFFFF-0000-1111-2222-333333333333}");
    };

    // Collaboration event callback types
    using CollaborationEventCallback = AZStd::function<void(const CollaborationEvent&)>;
    using UserJoinedCallback = AZStd::function<void(const UserPresence&)>;
    using UserLeftCallback = AZStd::function<void(const AZStd::string&)>;
    using ConflictDetectedCallback = AZStd::function<void(const CollaborationConflict&)>;
    using ConflictResolvedCallback = AZStd::function<void(const CollaborationConflict&)>;

    class CollaborationManager
    {
    public:
        CollaborationManager();
        ~CollaborationManager();

        // Session management
        bool CreateSession(const CollaborationSession& session);
        bool JoinSession(const AZStd::string& sessionId, const AZStd::string& password = "");
        bool LeaveSession();
        bool DestroySession(const AZStd::string& sessionId);
        CollaborationSession* GetCurrentSession();
        AZStd::vector<CollaborationSession> GetAvailableSessions();

        // User management
        bool AddUser(const UserPresence& user);
        bool RemoveUser(const AZStd::string& userId);
        bool UpdateUser(const UserPresence& user);
        UserPresence* GetUser(const AZStd::string& userId);
        AZStd::vector<UserPresence> GetAllUsers();
        int GetUserCount() const;

        // Cursor management
        bool UpdateUserCursor(const AZStd::string& userId, const UserCursor& cursor);
        UserCursor* GetUserCursor(const AZStd::string& userId);
        AZStd::vector<UserCursor> GetVisibleCursors();
        void SetCursorVisibility(const AZStd::string& userId, bool visible);

        // Edit lock management
        bool RequestEditLock(const AZStd::string& userId, const AZStd::string& nodeId, const AZStd::string& lockType = "exclusive");
        bool ReleaseEditLock(const AZStd::string& userId, const AZStd::string& nodeId);
        bool HasEditLock(const AZStd::string& userId, const AZStd::string& nodeId);
        AZStd::vector<UserEditLock> GetActiveEditLocks();
        void CleanupExpiredLocks();

        // Comment management
        AZStd::string AddComment(const CollaborationComment& comment);
        bool UpdateComment(const AZStd::string& commentId, const CollaborationComment& comment);
        bool DeleteComment(const AZStd::string& commentId);
        CollaborationComment* GetComment(const AZStd::string& commentId);
        AZStd::vector<CollaborationComment> GetCommentsForNode(const AZStd::string& nodeId);
        AZStd::vector<CollaborationComment> GetAllComments();

        // Voice and audio management
        bool StartVoiceChat(const AZStd::string& userId);
        bool StopVoiceChat(const AZStd::string& userId);
        bool SendVoiceData(const UserVoiceData& voiceData);
        bool IsVoiceChatActive(const AZStd::string& userId);
        void SetVoiceChatEnabled(bool enabled);
        bool IsVoiceChatEnabled() const;

        // Conflict resolution
        bool DetectConflict(const AZStd::string& nodeId, const AZStd::vector<AZStd::string>& involvedUsers);
        bool ResolveConflict(const AZStd::string& conflictId, const AZStd::string& resolutionStrategy);
        AZStd::vector<CollaborationConflict> GetActiveConflicts();
        CollaborationConflict* GetConflict(const AZStd::string& conflictId);

        // Permission management
        bool SetUserPermission(const AZStd::string& userId, UserPermission permission);
        UserPermission GetUserPermission(const AZStd::string& userId);
        bool HasPermission(const AZStd::string& userId, UserPermission requiredPermission);
        bool CanUserEdit(const AZStd::string& userId, const AZStd::string& nodeId);

        // Event broadcasting
        void BroadcastEvent(const CollaborationEvent& event);
        void BroadcastToUser(const AZStd::string& userId, const CollaborationEvent& event);
        void BroadcastToUsers(const AZStd::vector<AZStd::string>& userIds, const CollaborationEvent& event);

        // Network management
        bool InitializeNetwork(const AZStd::string& serverAddress, int port);
        bool ConnectToServer();
        bool DisconnectFromServer();
        bool IsConnected() const;
        float GetNetworkLatency() const;
        int GetBandwidthUsage() const;

        // Spatial audio integration
        bool InitializeSpatialAudio();
        void UpdateSpatialAudioPositions();
        void SetSpatialAudioEnabled(bool enabled);
        bool IsSpatialAudioEnabled() const;

        // Screen sharing
        bool StartScreenShare(const AZStd::string& userId);
        bool StopScreenShare(const AZStd::string& userId);
        bool IsScreenSharingActive(const AZStd::string& userId);
        void SetScreenShareEnabled(bool enabled);

        // File transfer
        bool StartFileTransfer(const AZStd::string& fromUserId, const AZStd::string& toUserId, const AZStd::string& filePath);
        bool AcceptFileTransfer(const AZStd::string& transferId);
        bool RejectFileTransfer(const AZStd::string& transferId);
        bool CancelFileTransfer(const AZStd::string& transferId);

        // Callback management
        void SetCollaborationEventCallback(CollaborationEventCallback callback);
        void SetUserJoinedCallback(UserJoinedCallback callback);
        void SetUserLeftCallback(UserLeftCallback callback);
        void SetConflictDetectedCallback(ConflictDetectedCallback callback);
        void SetConflictResolvedCallback(ConflictResolvedCallback callback);

        // Configuration
        void SetMaxUsers(int maxUsers);
        void SetSessionTimeout(float timeoutSeconds);
        void SetHeartbeatInterval(float intervalSeconds);
        void SetAutoReconnect(bool enabled);
        void SetEncryptionEnabled(bool enabled);
        void SetCompressionEnabled(bool enabled);

        // Statistics and monitoring
        int GetTotalMessages() const;
        int GetMessagesPerSecond() const;
        float GetAverageLatency() const;
        int GetBytesTransferred() const;
        AZStd::unordered_map<AZStd::string, int> GetUserActivityStats();

        // Debugging and diagnostics
        void EnableDebugMode(bool enabled);
        void SetLogLevel(int level);
        AZStd::string GetDebugInfo() const;
        void ExportSessionData(const AZStd::string& filePath);

        // Cleanup and shutdown
        void Cleanup();
        void Shutdown();

    private:
        // Internal state management
        void UpdateCollaborationState(float deltaTime);
        void ProcessIncomingMessages();
        void ProcessOutgoingMessages();
        void HandleUserTimeout();
        void HandleSessionTimeout();
        void HandleNetworkReconnection();

        // Message handling
        void HandleUserJoinedMessage(const AZStd::string& data);
        void HandleUserLeftMessage(const AZStd::string& data);
        void HandleCursorMoveMessage(const AZStd::string& data);
        void HandleNodeEditMessage(const AZStd::string& data);
        void HandleCommentMessage(const AZStd::string& data);
        void HandleVoiceDataMessage(const AZStd::string& data);
        void HandleConflictMessage(const AZStd::string& data);
        void HandlePermissionMessage(const AZStd::string& data);

        // Security and validation
        bool ValidateUser(const UserPresence& user);
        bool ValidateMessage(const CollaborationEvent& event);
        bool ValidatePermission(const AZStd::string& userId, const AZStd::string& action);
        void EncryptMessage(AZStd::string& message);
        void DecryptMessage(AZStd::string& message);

        // Optimization
        void OptimizeNetworkTraffic();
        void CompressMessageData(AZStd::string& data);
        void DecompressMessageData(AZStd::string& data);
        void PrioritizeMessages();

        // Error handling
        void HandleNetworkError(const AZStd::string& error);
        void HandleUserError(const AZStd::string& userId, const AZStd::string& error);
        void HandleSessionError(const AZStd::string& error);

        // Member variables
        AZStd::unique_ptr<NetworkManager> m_networkManager;
        AZStd::unique_ptr<SpatialAudioManager> m_spatialAudioManager;
        AZStd::unique_ptr<ConflictResolutionManager> m_conflictResolutionManager;
        AZStd::unique_ptr<CollaborationSecurityManager> m_securityManager;

        // Session state
        AZStd::unique_ptr<CollaborationSession> m_currentSession;
        AZStd::unordered_map<AZStd::string, UserPresence> m_users;
        AZStd::unordered_map<AZStd::string, UserCursor> m_userCursors;
        AZStd::unordered_map<AZStd::string, UserEditLock> m_editLocks;
        AZStd::unordered_map<AZStd::string, CollaborationComment> m_comments;
        AZStd::unordered_map<AZStd::string, CollaborationConflict> m_conflicts;
        AZStd::unordered_map<AZStd::string, UserPermission> m_userPermissions;

        // Callbacks
        CollaborationEventCallback m_eventCallback;
        UserJoinedCallback m_userJoinedCallback;
        UserLeftCallback m_userLeftCallback;
        ConflictDetectedCallback m_conflictDetectedCallback;
        ConflictResolvedCallback m_conflictResolvedCallback;

        // Configuration
        int m_maxUsers = 10;
        float m_sessionTimeout = 3600.0f; // 1 hour
        float m_heartbeatInterval = 30.0f; // 30 seconds
        bool m_autoReconnect = true;
        bool m_encryptionEnabled = true;
        bool m_compressionEnabled = true;
        bool m_voiceChatEnabled = true;
        bool m_spatialAudioEnabled = true;
        bool m_screenShareEnabled = true;
        bool m_fileTransferEnabled = true;

        // Statistics
        int m_totalMessages = 0;
        int m_messagesPerSecond = 0;
        float m_averageLatency = 0.0f;
        int m_bytesTransferred = 0;
        AZStd::unordered_map<AZStd::string, int> m_userActivityStats;

        // Threading and synchronization
        AZStd::mutex m_usersMutex;
        AZStd::mutex m_cursorsMutex;
        AZStd::mutex m_locksMutex;
        AZStd::mutex m_commentsMutex;
        AZStd::mutex m_conflictsMutex;
        AZStd::mutex m_messageQueueMutex;
        
        AZStd::vector<CollaborationEvent> m_incomingMessageQueue;
        AZStd::vector<CollaborationEvent> m_outgoingMessageQueue;
        
        AZStd::thread m_networkThread;
        AZStd::thread m_processingThread;
        bool m_isRunning = false;

        // Debug and logging
        bool m_debugMode = false;
        int m_logLevel = 2; // 0=None, 1=Error, 2=Warning, 3=Info, 4=Debug
        AZStd::string m_currentUserId;
        float m_lastUpdateTime = 0.0f;
        float m_lastHeartbeatTime = 0.0f;
    };

    // Utility functions
    AZStd::string GenerateUniqueId();
    AZStd::string GetCurrentTimestamp();
    AZ::Color GenerateUserColor(const AZStd::string& userId);
    bool IsValidSessionId(const AZStd::string& sessionId);
    bool IsValidUserId(const AZStd::string& userId);
    CollaborationEventType StringToEventType(const AZStd::string& eventType);
    AZStd::string EventTypeToString(CollaborationEventType eventType);
    UserPermission StringToPermission(const AZStd::string& permission);
    AZStd::string PermissionToString(UserPermission permission);

} // namespace Q3D 