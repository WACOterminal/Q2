#pragma once

#include <AzCore/std/containers/unordered_map.h>
#include <AzCore/std/containers/vector.h>
#include <AzCore/std/string/string.h>
#include <AzCore/std/smart_ptr/unique_ptr.h>
#include <AzCore/std/functional.h>
#include <AzCore/std/chrono/chrono.h>
#include <AzCore/std/parallel/mutex.h>
#include <AzCore/Component/TickBus.h>
#include <AzCore/Memory/SystemAllocator.h>

#include "WorkflowVisualization3DBus.h"
#include "WorkflowData.h"

namespace Q3D
{
    // Forward declarations
    class ScreenReaderInterface;
    class KeyboardNavigationManager;
    class VoiceCommandProcessor;
    class HapticFeedbackManager;
    class VisualAccessibilityManager;
    class CognitiveAccessibilityManager;
    class MotorAccessibilityManager;
    class AccessibilityProfileManager;

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

    enum class DisabilityType : int
    {
        Visual = 0,
        Hearing = 1,
        Motor = 2,
        Cognitive = 3,
        Speech = 4,
        Multiple = 5
    };

    enum class ColorBlindnessType : int
    {
        None = 0,
        Protanopia = 1,
        Deuteranopia = 2,
        Tritanopia = 3,
        Achromatopsia = 4
    };

    enum class MotorImpairmentType : int
    {
        None = 0,
        LimitedMobility = 1,
        Tremor = 2,
        LimitedReach = 3,
        OneHanded = 4,
        NoHands = 5
    };

    enum class CognitiveImpairmentType : int
    {
        None = 0,
        Dyslexia = 1,
        ADHD = 2,
        Autism = 3,
        MemoryImpairment = 4,
        LearningDisabilities = 5
    };

    enum class NavigationMode : int
    {
        Mouse = 0,
        Keyboard = 1,
        Voice = 2,
        Gesture = 3,
        EyeTracking = 4,
        Switch = 5,
        Joystick = 6,
        TouchScreen = 7,
        HeadMovement = 8,
        BrainInterface = 9
    };

    enum class FeedbackType : int
    {
        Visual = 0,
        Audio = 1,
        Haptic = 2,
        Braille = 3,
        Voice = 4,
        Vibration = 5
    };

    struct AccessibilityProfile
    {
        AZStd::string profileId;
        AZStd::string name;
        AZStd::string description;
        AZStd::string userId;
        
        // Disability information
        AZStd::vector<DisabilityType> disabilityTypes;
        ColorBlindnessType colorBlindness = ColorBlindnessType::None;
        MotorImpairmentType motorImpairment = MotorImpairmentType::None;
        CognitiveImpairmentType cognitiveImpairment = CognitiveImpairmentType::None;
        
        // Enabled features
        AZStd::unordered_map<AccessibilityFeature, bool> enabledFeatures;
        
        // Navigation preferences
        NavigationMode primaryNavigationMode = NavigationMode::Mouse;
        NavigationMode fallbackNavigationMode = NavigationMode::Keyboard;
        
        // Feedback preferences
        AZStd::vector<FeedbackType> preferredFeedbackTypes;
        
        // Visual settings
        bool highContrastMode = false;
        bool reducedMotion = false;
        bool flashingContentFiltering = false;
        float fontScaleFactor = 1.0f;
        float uiScaleFactor = 1.0f;
        bool forceFocus = false;
        bool enhancedFocusIndicators = false;
        AZStd::string preferredColorScheme = "default";
        
        // Audio settings
        bool screenReaderEnabled = false;
        bool audioDescriptionsEnabled = false;
        bool soundEffectsEnabled = true;
        bool spatialAudioEnabled = false;
        float audioVolume = 1.0f;
        float speechRate = 1.0f;
        float speechPitch = 1.0f;
        AZStd::string preferredVoice = "default";
        
        // Motor settings
        bool stickyKeys = false;
        bool slowKeys = false;
        bool bounceKeys = false;
        bool mouseKeys = false;
        float clickHoldDuration = 0.5f;
        float doubleClickTime = 0.5f;
        float mouseSpeed = 1.0f;
        int dwellClickTime = 1000; // milliseconds
        
        // Cognitive settings
        bool simplifiedInterface = false;
        bool cognitiveAids = false;
        bool extendedTimeouts = false;
        bool progressIndicators = true;
        bool confirmationDialogs = false;
        bool autoSave = true;
        bool readingGuides = false;
        bool wordPrediction = false;
        
        // Interaction settings
        bool allowVoiceCommands = false;
        bool allowGestureControls = false;
        bool allowEyeTracking = false;
        bool allowSwitchControl = false;
        bool allowBrailleInput = false;
        bool allowCustomControls = false;
        
        // Timing settings
        float animationSpeed = 1.0f;
        float transitionSpeed = 1.0f;
        float autoAdvanceTime = 0.0f; // 0 = disabled
        float messageDisplayTime = 5.0f;
        float tooltipDelay = 1.0f;
        
        // Custom settings
        AZStd::unordered_map<AZStd::string, AZStd::string> customSettings;
        
        AZ_TYPE_INFO(AccessibilityProfile, "{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE}");
    };

    struct AccessibilitySettings
    {
        AccessibilityLevel level = AccessibilityLevel::Basic;
        AZStd::string activeProfileId;
        
        // Global settings
        bool enableGlobalShortcuts = true;
        bool enableAutoDetection = true;
        bool enableTelemetry = false;
        bool enableDebugMode = false;
        
        // Screen reader settings
        bool enableScreenReaderSupport = true;
        AZStd::string screenReaderType = "NVDA"; // NVDA, JAWS, VoiceOver, etc.
        bool announceSystemMessages = true;
        bool announcePageChanges = true;
        bool announceMenuItems = true;
        bool announceButtonStates = true;
        
        // Keyboard navigation settings
        bool enableKeyboardNavigation = true;
        bool enableTabNavigation = true;
        bool enableArrowKeyNavigation = true;
        bool enableShortcutKeys = true;
        bool enableEscapeKey = true;
        bool enableEnterKey = true;
        bool enableSpaceKey = true;
        bool skipToContentEnabled = true;
        
        // Voice command settings
        bool enableVoiceCommands = false;
        float voiceCommandThreshold = 0.8f;
        AZStd::string voiceCommandLanguage = "en-US";
        bool enableVoiceNavigation = false;
        bool enableVoiceAnnotation = false;
        
        // Haptic feedback settings
        bool enableHapticFeedback = false;
        float hapticIntensity = 1.0f;
        bool enableTactileFeedback = false;
        bool enableVibrationFeedback = false;
        
        // Visual accessibility settings
        bool enableHighContrast = false;
        bool enableColorBlindSupport = false;
        bool enableMotionReduction = false;
        bool enableFontScaling = false;
        bool enableFocusIndicators = true;
        bool enableSkipLinks = true;
        bool enableLiveRegions = true;
        
        // Audio accessibility settings
        bool enableAudioDescriptions = false;
        bool enableCaptionsSubtitles = false;
        bool enableSpatialAudio = false;
        bool enableAudioCues = true;
        
        // Motor accessibility settings
        bool enableGestureAlternatives = false;
        bool enableSlowMotionMode = false;
        bool enableSwitchControl = false;
        bool enableEyeTracking = false;
        bool enableHeadTracking = false;
        
        // Cognitive accessibility settings
        bool enableSimplifiedInterface = false;
        bool enableCognitiveAids = false;
        bool enableReadingSupport = false;
        bool enableMemoryAids = false;
        bool enableAttentionSupport = false;
        
        // Custom accessibility settings
        AZStd::unordered_map<AZStd::string, bool> customFeatures;
        AZStd::unordered_map<AZStd::string, float> customValues;
        AZStd::unordered_map<AZStd::string, AZStd::string> customStrings;
        
        AZ_TYPE_INFO(AccessibilitySettings, "{BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF}");
    };

    struct AccessibilityEvent
    {
        AZStd::string eventId;
        AZStd::string eventType;
        AZStd::string elementId;
        AZStd::string elementType;
        AZStd::string elementLabel;
        AZStd::string elementDescription;
        AZStd::string elementValue;
        AZStd::string elementRole;
        AZStd::string elementState;
        AZStd::string action;
        AZStd::string context;
        float timestamp;
        AZStd::unordered_map<AZStd::string, AZStd::string> additionalData;
        
        AZ_TYPE_INFO(AccessibilityEvent, "{CCCCCCCC-DDDD-EEEE-FFFF-000000000000}");
    };

    struct AccessibilityAlert
    {
        AZStd::string alertId;
        AZStd::string alertType;
        AZStd::string message;
        AZStd::string description;
        AZStd::string recommendation;
        AccessibilityLevel severity;
        float timestamp;
        bool isActive = true;
        bool hasBeenAnnounced = false;
        
        AZ_TYPE_INFO(AccessibilityAlert, "{DDDDDDDD-EEEE-FFFF-0000-111111111111}");
    };

    struct FocusableElement
    {
        AZStd::string elementId;
        AZStd::string elementType;
        AZStd::string label;
        AZStd::string description;
        AZStd::string role;
        AZStd::string value;
        AZStd::string state;
        WorkflowNode* associatedNode = nullptr;
        AZ::Vector3 position;
        AZ::Vector3 size;
        bool isVisible = true;
        bool isEnabled = true;
        bool isFocusable = true;
        bool isKeyboardAccessible = true;
        int tabOrder = 0;
        
        AZ_TYPE_INFO(FocusableElement, "{EEEEEEEE-FFFF-0000-1111-222222222222}");
    };

    struct KeyboardShortcut
    {
        AZStd::string shortcutId;
        AZStd::string name;
        AZStd::string description;
        AZStd::string keySequence;
        AZStd::string action;
        AZStd::string context;
        bool isGlobal = false;
        bool isEnabled = true;
        bool isCustomizable = true;
        
        AZ_TYPE_INFO(KeyboardShortcut, "{FFFFFFFF-0000-1111-2222-333333333333}");
    };

    struct VoiceCommand
    {
        AZStd::string commandId;
        AZStd::string name;
        AZStd::string description;
        AZStd::vector<AZStd::string> triggerPhrases;
        AZStd::string action;
        AZStd::string context;
        float confidence = 0.8f;
        bool isEnabled = true;
        bool isCustomizable = true;
        
        AZ_TYPE_INFO(VoiceCommand, "{00000000-1111-2222-3333-444444444444}");
    };

    // Callback types
    using AccessibilityEventCallback = AZStd::function<void(const AccessibilityEvent&)>;
    using AccessibilityAlertCallback = AZStd::function<void(const AccessibilityAlert&)>;
    using FocusChangedCallback = AZStd::function<void(const FocusableElement&)>;
    using ScreenReaderCallback = AZStd::function<void(const AZStd::string&)>;
    using VoiceCommandCallback = AZStd::function<void(const VoiceCommand&)>;

    class AccessibilityManager
    {
    public:
        AccessibilityManager();
        ~AccessibilityManager();

        // Initialization and lifecycle
        bool Initialize();
        void Shutdown();
        bool IsInitialized() const;
        void Update(float deltaTime);

        // Profile management
        bool LoadAccessibilityProfile(const AZStd::string& profileId);
        bool SaveAccessibilityProfile(const AccessibilityProfile& profile);
        AccessibilityProfile* GetCurrentProfile();
        AZStd::vector<AccessibilityProfile> GetAvailableProfiles() const;
        AccessibilityProfile CreateDefaultProfile();
        AccessibilityProfile CreateProfileForDisability(DisabilityType disabilityType);
        bool DeleteProfile(const AZStd::string& profileId);

        // Settings management
        void SetAccessibilitySettings(const AccessibilitySettings& settings);
        AccessibilitySettings GetAccessibilitySettings() const;
        void SetAccessibilityLevel(AccessibilityLevel level);
        AccessibilityLevel GetAccessibilityLevel() const;
        bool IsFeatureEnabled(AccessibilityFeature feature) const;
        void SetFeatureEnabled(AccessibilityFeature feature, bool enabled);

        // Screen reader support
        void EnableScreenReader(bool enabled);
        bool IsScreenReaderEnabled() const;
        void AnnounceText(const AZStd::string& text, bool interrupt = false);
        void AnnounceElementFocus(const FocusableElement& element);
        void AnnounceElementChange(const FocusableElement& element);
        void AnnounceSystemMessage(const AZStd::string& message);
        void AnnouncePageChange(const AZStd::string& pageTitle);
        void AnnounceMenuNavigation(const AZStd::string& menuItem);
        void AnnounceButtonState(const AZStd::string& buttonId, bool pressed);
        void AnnounceProgressUpdate(const AZStd::string& operation, float progress);
        void AnnounceError(const AZStd::string& errorMessage);
        void AnnounceSuccess(const AZStd::string& successMessage);

        // Keyboard navigation
        void EnableKeyboardNavigation(bool enabled);
        bool IsKeyboardNavigationEnabled() const;
        void SetFocusedElement(const AZStd::string& elementId);
        FocusableElement* GetFocusedElement();
        AZStd::string GetFocusedElementId() const;
        bool MoveFocusNext();
        bool MoveFocusPrevious();
        bool MoveFocusUp();
        bool MoveFocusDown();
        bool MoveFocusLeft();
        bool MoveFocusRight();
        bool MoveFocusToFirst();
        bool MoveFocusToLast();
        bool ActivateFocusedElement();
        void RegisterFocusableElement(const FocusableElement& element);
        void UnregisterFocusableElement(const AZStd::string& elementId);
        void UpdateFocusableElement(const FocusableElement& element);
        AZStd::vector<FocusableElement> GetFocusableElements() const;

        // Keyboard shortcuts
        void RegisterKeyboardShortcut(const KeyboardShortcut& shortcut);
        void UnregisterKeyboardShortcut(const AZStd::string& shortcutId);
        bool HandleKeyboardInput(const AZStd::string& keySequence);
        AZStd::vector<KeyboardShortcut> GetRegisteredShortcuts() const;
        AZStd::vector<KeyboardShortcut> GetShortcutsForContext(const AZStd::string& context) const;
        void SetShortcutEnabled(const AZStd::string& shortcutId, bool enabled);
        void CustomizeShortcut(const AZStd::string& shortcutId, const AZStd::string& newKeySequence);

        // Voice commands
        void EnableVoiceCommands(bool enabled);
        bool IsVoiceCommandsEnabled() const;
        void RegisterVoiceCommand(const VoiceCommand& command);
        void UnregisterVoiceCommand(const AZStd::string& commandId);
        bool ProcessVoiceInput(const AZStd::string& speechText, float confidence);
        AZStd::vector<VoiceCommand> GetRegisteredVoiceCommands() const;
        AZStd::vector<VoiceCommand> GetVoiceCommandsForContext(const AZStd::string& context) const;
        void SetVoiceCommandEnabled(const AZStd::string& commandId, bool enabled);
        void SetVoiceCommandConfidenceThreshold(float threshold);
        float GetVoiceCommandConfidenceThreshold() const;

        // Haptic feedback
        void EnableHapticFeedback(bool enabled);
        bool IsHapticFeedbackEnabled() const;
        void TriggerHapticFeedback(const AZStd::string& feedbackType, float intensity = 1.0f);
        void TriggerTactileFeedback(const AZStd::string& pattern);
        void TriggerVibrationFeedback(int duration, float intensity);
        void SetHapticIntensity(float intensity);
        float GetHapticIntensity() const;

        // Visual accessibility
        void EnableHighContrast(bool enabled);
        bool IsHighContrastEnabled() const;
        void EnableColorBlindSupport(bool enabled);
        bool IsColorBlindSupportEnabled() const;
        void SetColorBlindnessType(ColorBlindnessType type);
        ColorBlindnessType GetColorBlindnessType() const;
        void EnableMotionReduction(bool enabled);
        bool IsMotionReductionEnabled() const;
        void SetFontScaleFactor(float scaleFactor);
        float GetFontScaleFactor() const;
        void SetUIScaleFactor(float scaleFactor);
        float GetUIScaleFactor() const;
        void EnableFocusIndicators(bool enabled);
        bool IsFocusIndicatorsEnabled() const;
        void SetPreferredColorScheme(const AZStd::string& scheme);
        AZStd::string GetPreferredColorScheme() const;

        // Audio accessibility
        void EnableAudioDescriptions(bool enabled);
        bool IsAudioDescriptionsEnabled() const;
        void EnableCaptionsSubtitles(bool enabled);
        bool IsCaptionsSubtitlesEnabled() const;
        void EnableSpatialAudio(bool enabled);
        bool IsSpatialAudioEnabled() const;
        void SetAudioVolume(float volume);
        float GetAudioVolume() const;
        void SetSpeechRate(float rate);
        float GetSpeechRate() const;
        void SetSpeechPitch(float pitch);
        float GetSpeechPitch() const;
        void SetPreferredVoice(const AZStd::string& voice);
        AZStd::string GetPreferredVoice() const;

        // Motor accessibility
        void EnableGestureAlternatives(bool enabled);
        bool IsGestureAlternativesEnabled() const;
        void EnableSlowMotionMode(bool enabled);
        bool IsSlowMotionModeEnabled() const;
        void EnableSwitchControl(bool enabled);
        bool IsSwitchControlEnabled() const;
        void EnableEyeTracking(bool enabled);
        bool IsEyeTrackingEnabled() const;
        void SetClickHoldDuration(float duration);
        float GetClickHoldDuration() const;
        void SetDoubleClickTime(float time);
        float GetDoubleClickTime() const;
        void SetMouseSpeed(float speed);
        float GetMouseSpeed() const;
        void SetDwellClickTime(int milliseconds);
        int GetDwellClickTime() const;

        // Cognitive accessibility
        void EnableSimplifiedInterface(bool enabled);
        bool IsSimplifiedInterfaceEnabled() const;
        void EnableCognitiveAids(bool enabled);
        bool IsCognitiveAidsEnabled() const;
        void EnableExtendedTimeouts(bool enabled);
        bool IsExtendedTimeoutsEnabled() const;
        void EnableProgressIndicators(bool enabled);
        bool IsProgressIndicatorsEnabled() const;
        void EnableConfirmationDialogs(bool enabled);
        bool IsConfirmationDialogsEnabled() const;
        void EnableAutoSave(bool enabled);
        bool IsAutoSaveEnabled() const;
        void EnableReadingGuides(bool enabled);
        bool IsReadingGuidesEnabled() const;
        void EnableWordPrediction(bool enabled);
        bool IsWordPredictionEnabled() const;

        // Timing and animation
        void SetAnimationSpeed(float speed);
        float GetAnimationSpeed() const;
        void SetTransitionSpeed(float speed);
        float GetTransitionSpeed() const;
        void SetAutoAdvanceTime(float time);
        float GetAutoAdvanceTime() const;
        void SetMessageDisplayTime(float time);
        float GetMessageDisplayTime() const;
        void SetTooltipDelay(float delay);
        float GetTooltipDelay() const;

        // Event handling
        void HandleAccessibilityEvent(const AccessibilityEvent& event);
        void TriggerAccessibilityAlert(const AccessibilityAlert& alert);
        AZStd::vector<AccessibilityAlert> GetActiveAlerts() const;
        void AcknowledgeAlert(const AZStd::string& alertId);
        void ClearAlert(const AZStd::string& alertId);
        void ClearAllAlerts();

        // Workflow-specific accessibility
        void MakeWorkflowAccessible(const WorkflowData& workflow);
        void AnnounceWorkflowChange(const WorkflowData& workflow);
        void AnnounceNodeFocus(const WorkflowNode& node);
        void AnnounceNodeActivation(const WorkflowNode& node);
        void AnnounceConnectionCreated(const WorkflowNode& fromNode, const WorkflowNode& toNode);
        void AnnounceConnectionRemoved(const WorkflowNode& fromNode, const WorkflowNode& toNode);
        void AnnounceWorkflowExecution(const WorkflowData& workflow);
        void AnnounceExecutionProgress(const WorkflowData& workflow, float progress);
        void AnnounceExecutionComplete(const WorkflowData& workflow);
        void AnnounceExecutionError(const WorkflowData& workflow, const AZStd::string& error);

        // Live regions
        void CreateLiveRegion(const AZStd::string& regionId, const AZStd::string& regionType);
        void UpdateLiveRegion(const AZStd::string& regionId, const AZStd::string& content);
        void RemoveLiveRegion(const AZStd::string& regionId);
        AZStd::vector<AZStd::string> GetLiveRegions() const;

        // Skip links
        void CreateSkipLink(const AZStd::string& linkId, const AZStd::string& targetId, const AZStd::string& label);
        void RemoveSkipLink(const AZStd::string& linkId);
        bool ActivateSkipLink(const AZStd::string& linkId);
        AZStd::vector<AZStd::string> GetSkipLinks() const;

        // Landmark navigation
        void CreateLandmark(const AZStd::string& landmarkId, const AZStd::string& landmarkType, const AZStd::string& label);
        void RemoveLandmark(const AZStd::string& landmarkId);
        bool NavigateToLandmark(const AZStd::string& landmarkId);
        AZStd::vector<AZStd::string> GetLandmarks() const;

        // Alternative text
        void SetAlternativeText(const AZStd::string& elementId, const AZStd::string& altText);
        AZStd::string GetAlternativeText(const AZStd::string& elementId) const;
        void SetElementDescription(const AZStd::string& elementId, const AZStd::string& description);
        AZStd::string GetElementDescription(const AZStd::string& elementId) const;

        // ARIA attributes
        void SetAriaLabel(const AZStd::string& elementId, const AZStd::string& label);
        AZStd::string GetAriaLabel(const AZStd::string& elementId) const;
        void SetAriaRole(const AZStd::string& elementId, const AZStd::string& role);
        AZStd::string GetAriaRole(const AZStd::string& elementId) const;
        void SetAriaState(const AZStd::string& elementId, const AZStd::string& state, const AZStd::string& value);
        AZStd::string GetAriaState(const AZStd::string& elementId, const AZStd::string& state) const;
        void SetAriaProperty(const AZStd::string& elementId, const AZStd::string& property, const AZStd::string& value);
        AZStd::string GetAriaProperty(const AZStd::string& elementId, const AZStd::string& property) const;

        // Callbacks
        void SetAccessibilityEventCallback(AccessibilityEventCallback callback);
        void SetAccessibilityAlertCallback(AccessibilityAlertCallback callback);
        void SetFocusChangedCallback(FocusChangedCallback callback);
        void SetScreenReaderCallback(ScreenReaderCallback callback);
        void SetVoiceCommandCallback(VoiceCommandCallback callback);

        // Testing and validation
        bool ValidateAccessibility();
        AZStd::vector<AZStd::string> GetAccessibilityIssues() const;
        AZStd::string GenerateAccessibilityReport() const;
        void RunAccessibilityAudit();
        float GetAccessibilityScore() const;

        // Platform integration
        bool DetectAssistiveTechnology();
        AZStd::vector<AZStd::string> GetDetectedAssistiveTechnology() const;
        bool IsScreenReaderActive() const;
        bool IsVoiceControlActive() const;
        bool IsHighContrastActive() const;
        bool IsKeyboardNavigationActive() const;

        // Custom accessibility extensions
        void RegisterCustomAccessibilityFeature(const AZStd::string& featureId, const AZStd::string& featureName);
        void UnregisterCustomAccessibilityFeature(const AZStd::string& featureId);
        void EnableCustomAccessibilityFeature(const AZStd::string& featureId, bool enabled);
        bool IsCustomAccessibilityFeatureEnabled(const AZStd::string& featureId) const;

        // Configuration
        void LoadConfiguration(const AZStd::string& configPath);
        void SaveConfiguration(const AZStd::string& configPath);
        void ResetToDefaults();
        bool IsConfigurationValid() const;

        // Debug and logging
        void EnableDebugMode(bool enabled);
        bool IsDebugModeEnabled() const;
        void SetLogLevel(int level);
        int GetLogLevel() const;
        void LogAccessibilityEvent(const AccessibilityEvent& event);
        void LogAccessibilityIssue(const AZStd::string& issue);

        // Statistics
        int GetTotalAccessibilityEvents() const;
        int GetFocusChangeCount() const;
        int GetVoiceCommandCount() const;
        int GetKeyboardShortcutCount() const;
        float GetAverageScreenReaderUsage() const;
        AZStd::unordered_map<AZStd::string, int> GetFeatureUsageStatistics() const;

    private:
        // Internal managers
        void InitializeSubManagers();
        void ShutdownSubManagers();
        void UpdateSubManagers(float deltaTime);

        // Profile helpers
        void ApplyAccessibilityProfile(const AccessibilityProfile& profile);
        void ResetAccessibilitySettings();
        AccessibilityProfile CreateOptimalProfile(const AZStd::vector<DisabilityType>& disabilities);

        // Screen reader helpers
        void InitializeScreenReaderSupport();
        void ShutdownScreenReaderSupport();
        void UpdateScreenReaderSupport();
        AZStd::string PrepareTextForScreenReader(const AZStd::string& text) const;
        void QueueScreenReaderMessage(const AZStd::string& message, bool interrupt);

        // Keyboard navigation helpers
        void InitializeKeyboardNavigation();
        void ShutdownKeyboardNavigation();
        void UpdateKeyboardNavigation();
        void BuildFocusChain();
        void UpdateFocusIndicators();
        FocusableElement* FindNextFocusableElement(const FocusableElement* current) const;
        FocusableElement* FindPreviousFocusableElement(const FocusableElement* current) const;

        // Voice command helpers
        void InitializeVoiceCommands();
        void ShutdownVoiceCommands();
        void UpdateVoiceCommands();
        VoiceCommand* FindMatchingVoiceCommand(const AZStd::string& speechText, float confidence) const;
        void ExecuteVoiceCommand(const VoiceCommand& command);

        // Visual accessibility helpers
        void InitializeVisualAccessibility();
        void ShutdownVisualAccessibility();
        void UpdateVisualAccessibility();
        void ApplyHighContrastMode();
        void ApplyColorBlindSupport();
        void ApplyMotionReduction();
        void ApplyFontScaling();
        void ApplyUIScaling();

        // Audio accessibility helpers
        void InitializeAudioAccessibility();
        void ShutdownAudioAccessibility();
        void UpdateAudioAccessibility();
        void ConfigureAudioSettings();
        void ConfigureSpeechSettings();

        // Motor accessibility helpers
        void InitializeMotorAccessibility();
        void ShutdownMotorAccessibility();
        void UpdateMotorAccessibility();
        void ConfigureMotorSettings();
        void ProcessMotorInput();

        // Cognitive accessibility helpers
        void InitializeCognitiveAccessibility();
        void ShutdownCognitiveAccessibility();
        void UpdateCognitiveAccessibility();
        void ApplyCognitiveSettings();
        void UpdateCognitiveAids();

        // Event processing
        void ProcessAccessibilityEvents();
        void ProcessFocusChanges();
        void ProcessKeyboardInput();
        void ProcessVoiceInput();
        void ProcessHapticFeedback();

        // Validation helpers
        bool ValidateElementAccessibility(const FocusableElement& element) const;
        bool ValidateKeyboardAccessibility() const;
        bool ValidateScreenReaderCompatibility() const;
        bool ValidateColorContrast() const;
        bool ValidateTextAlternatives() const;
        bool ValidateARIAAttributes() const;
        AZStd::vector<AZStd::string> GetAccessibilityViolations() const;

        // Platform detection
        void DetectPlatformCapabilities();
        void DetectSystemAccessibilitySettings();
        void DetectInstalledAssistiveTechnology();
        void SynchronizeWithSystemSettings();

        // Data management
        void LoadAccessibilityData();
        void SaveAccessibilityData();
        void ClearAccessibilityData();
        void ValidateAccessibilityData();

        // Member variables
        AZStd::unique_ptr<ScreenReaderInterface> m_screenReader;
        AZStd::unique_ptr<KeyboardNavigationManager> m_keyboardNavigation;
        AZStd::unique_ptr<VoiceCommandProcessor> m_voiceCommands;
        AZStd::unique_ptr<HapticFeedbackManager> m_hapticFeedback;
        AZStd::unique_ptr<VisualAccessibilityManager> m_visualAccessibility;
        AZStd::unique_ptr<CognitiveAccessibilityManager> m_cognitiveAccessibility;
        AZStd::unique_ptr<MotorAccessibilityManager> m_motorAccessibility;
        AZStd::unique_ptr<AccessibilityProfileManager> m_profileManager;

        // State
        bool m_isInitialized = false;
        bool m_isActive = false;
        bool m_debugMode = false;
        int m_logLevel = 2;
        AccessibilityLevel m_currentLevel = AccessibilityLevel::Basic;

        // Current profile and settings
        AZStd::unique_ptr<AccessibilityProfile> m_currentProfile;
        AccessibilitySettings m_currentSettings;

        // Focusable elements
        AZStd::unordered_map<AZStd::string, FocusableElement> m_focusableElements;
        AZStd::vector<AZStd::string> m_focusChain;
        AZStd::string m_focusedElementId;

        // Keyboard shortcuts
        AZStd::unordered_map<AZStd::string, KeyboardShortcut> m_keyboardShortcuts;
        AZStd::unordered_map<AZStd::string, AZStd::vector<AZStd::string>> m_contextShortcuts;

        // Voice commands
        AZStd::unordered_map<AZStd::string, VoiceCommand> m_voiceCommands;
        AZStd::unordered_map<AZStd::string, AZStd::vector<AZStd::string>> m_contextVoiceCommands;

        // Live regions and landmarks
        AZStd::unordered_map<AZStd::string, AZStd::string> m_liveRegions;
        AZStd::unordered_map<AZStd::string, AZStd::string> m_landmarks;
        AZStd::unordered_map<AZStd::string, AZStd::string> m_skipLinks;

        // Alternative text and descriptions
        AZStd::unordered_map<AZStd::string, AZStd::string> m_alternativeTexts;
        AZStd::unordered_map<AZStd::string, AZStd::string> m_elementDescriptions;

        // ARIA attributes
        AZStd::unordered_map<AZStd::string, AZStd::unordered_map<AZStd::string, AZStd::string>> m_ariaAttributes;

        // Alerts and events
        AZStd::unordered_map<AZStd::string, AccessibilityAlert> m_activeAlerts;
        AZStd::vector<AccessibilityEvent> m_eventQueue;

        // Callbacks
        AccessibilityEventCallback m_eventCallback;
        AccessibilityAlertCallback m_alertCallback;
        FocusChangedCallback m_focusChangedCallback;
        ScreenReaderCallback m_screenReaderCallback;
        VoiceCommandCallback m_voiceCommandCallback;

        // Statistics
        int m_totalAccessibilityEvents = 0;
        int m_focusChangeCount = 0;
        int m_voiceCommandCount = 0;
        int m_keyboardShortcutCount = 0;
        float m_screenReaderUsageTime = 0.0f;
        AZStd::unordered_map<AZStd::string, int> m_featureUsageStats;

        // Platform information
        AZStd::vector<AZStd::string> m_detectedAssistiveTechnology;
        AZStd::unordered_map<AZStd::string, bool> m_platformCapabilities;
        AZStd::unordered_map<AZStd::string, AZStd::string> m_systemSettings;

        // Custom features
        AZStd::unordered_map<AZStd::string, AZStd::string> m_customFeatures;
        AZStd::unordered_map<AZStd::string, bool> m_customFeatureStates;

        // Timing
        AZStd::chrono::steady_clock::time_point m_startTime;
        AZStd::chrono::steady_clock::time_point m_lastUpdateTime;

        // Threading
        mutable AZStd::mutex m_accessibilityMutex;
        mutable AZStd::mutex m_focusMutex;
        mutable AZStd::mutex m_eventMutex;
        mutable AZStd::mutex m_alertMutex;
    };

    // Utility functions
    AZStd::string AccessibilityFeatureToString(AccessibilityFeature feature);
    AccessibilityFeature StringToAccessibilityFeature(const AZStd::string& featureString);
    AZStd::string AccessibilityLevelToString(AccessibilityLevel level);
    AccessibilityLevel StringToAccessibilityLevel(const AZStd::string& levelString);
    AZStd::string DisabilityTypeToString(DisabilityType type);
    DisabilityType StringToDisabilityType(const AZStd::string& typeString);
    AZStd::string NavigationModeToString(NavigationMode mode);
    NavigationMode StringToNavigationMode(const AZStd::string& modeString);
    AZStd::string FeedbackTypeToString(FeedbackType type);
    FeedbackType StringToFeedbackType(const AZStd::string& typeString);

    // Accessibility validation utilities
    bool ValidateColorContrast(const AZStd::string& foregroundColor, const AZStd::string& backgroundColor);
    bool ValidateTextSize(float fontSize, float viewingDistance);
    bool ValidateClickTargetSize(float width, float height);
    bool ValidateKeyboardAccessibility(const FocusableElement& element);
    bool ValidateScreenReaderCompatibility(const FocusableElement& element);
    AZStd::vector<AZStd::string> GetAccessibilityRecommendations(const FocusableElement& element);

    // Color transformation utilities
    AZStd::string ApplyColorBlindnessFilter(const AZStd::string& color, ColorBlindnessType type);
    AZStd::string ApplyHighContrastFilter(const AZStd::string& color, bool isBackground);
    float CalculateColorContrast(const AZStd::string& color1, const AZStd::string& color2);
    AZStd::string GetHighContrastColor(const AZStd::string& originalColor);

} // namespace Q3D 