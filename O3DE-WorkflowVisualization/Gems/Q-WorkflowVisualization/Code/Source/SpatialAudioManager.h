#pragma once

#include <AzCore/std/containers/unordered_map.h>
#include <AzCore/std/containers/vector.h>
#include <AzCore/std/string/string.h>
#include <AzCore/std/smart_ptr/unique_ptr.h>
#include <AzCore/std/functional.h>
#include <AzCore/Math/Vector3.h>
#include <AzCore/Math/Quaternion.h>
#include <AzCore/std/parallel/mutex.h>
#include <AzCore/std/chrono/chrono.h>
#include <AzCore/Component/TickBus.h>

#include "WorkflowVisualization3DBus.h"

namespace Q3D
{
    // Forward declarations
    class AudioSource;
    class AudioListener;
    class AudioProcessor;
    class AudioEffectsProcessor;
    class VoiceActivityDetector;
    class NoiseSuppressionFilter;
    class EchoCancellationFilter;

    enum class AudioCodec : int
    {
        PCM = 0,
        Opus = 1,
        AAC = 2,
        G711 = 3,
        G722 = 4,
        Speex = 5
    };

    enum class SpatialAudioModel : int
    {
        HRTF = 0,          // Head-Related Transfer Function
        Binaural = 1,      // Binaural processing
        Ambisonics = 2,    // Ambisonic spatial audio
        SimpleDistance = 3  // Simple distance-based attenuation
    };

    enum class ReverbPreset : int
    {
        None = 0,
        Room = 1,
        Hall = 2,
        Cathedral = 3,
        Cave = 4,
        Underwater = 5,
        Space = 6,
        Custom = 7
    };

    struct AudioSettings
    {
        // Basic settings
        int sampleRate = 44100;
        int channels = 2;
        int bitsPerSample = 16;
        int bufferSize = 512;
        AudioCodec codec = AudioCodec::Opus;
        
        // Spatial audio settings
        SpatialAudioModel spatialModel = SpatialAudioModel::HRTF;
        float maxDistance = 100.0f;
        float referenceDistance = 1.0f;
        float rolloffFactor = 1.0f;
        float dopplerFactor = 1.0f;
        float speedOfSound = 343.0f; // m/s
        
        // Quality settings
        int bitrate = 64000; // bits per second
        int complexity = 10; // 0-10 for Opus
        bool enableDTX = true; // Discontinuous Transmission
        bool enableFEC = true; // Forward Error Correction
        
        // Processing settings
        bool enableVAD = true; // Voice Activity Detection
        bool enableNoiseSuppression = true;
        bool enableEchoCancellation = true;
        bool enableAutomaticGainControl = true;
        
        // Spatial effects
        bool enableReverb = true;
        ReverbPreset reverbPreset = ReverbPreset::Room;
        float reverbAmount = 0.3f;
        bool enableOcclusion = true;
        float occlusionFactor = 0.5f;
        
        AZ_TYPE_INFO(AudioSettings, "{11111111-2222-3333-4444-555555555555}");
    };

    struct UserAudioProfile
    {
        AZStd::string userId;
        AZStd::string displayName;
        
        // Spatial properties
        AZ::Vector3 position;
        AZ::Vector3 velocity;
        AZ::Quaternion orientation;
        float gain = 1.0f;
        bool muted = false;
        bool localMuted = false;
        
        // Voice properties
        float pitch = 1.0f;
        float timbre = 1.0f;
        AZ::Vector3 voiceColor = AZ::Vector3(1.0f, 1.0f, 1.0f);
        
        // Audio source properties
        bool isTransmitting = false;
        float transmissionQuality = 1.0f;
        float networkLatency = 0.0f;
        float packetLoss = 0.0f;
        
        // Environmental properties
        float occlusionLevel = 0.0f;
        float obstructionLevel = 0.0f;
        float reverbLevel = 1.0f;
        
        // Activity tracking
        float vadLevel = 0.0f;
        float speechLevel = 0.0f;
        float noiseLevel = 0.0f;
        float lastActivityTime = 0.0f;
        
        AZ_TYPE_INFO(UserAudioProfile, "{22222222-3333-4444-5555-666666666666}");
    };

    struct AudioEnvironment
    {
        AZStd::string environmentId;
        AZStd::string name;
        
        // Acoustic properties
        float roomSize = 10.0f;
        float absorptionCoefficient = 0.3f;
        float reflectivity = 0.7f;
        float decayTime = 1.5f;
        float diffusion = 0.5f;
        
        // Reverb properties
        ReverbPreset reverbPreset = ReverbPreset::Room;
        float reverbTime = 1.0f;
        float reverbDamping = 0.5f;
        float reverbLevel = 0.3f;
        
        // Environmental effects
        float airAbsorption = 0.01f;
        float temperature = 20.0f; // Celsius
        float humidity = 50.0f;    // Percentage
        float pressure = 1013.25f; // hPa
        
        // Occlusion geometry
        AZStd::vector<AZ::Vector3> occludingGeometry;
        AZStd::vector<float> materialProperties;
        
        AZ_TYPE_INFO(AudioEnvironment, "{33333333-4444-5555-6666-777777777777}");
    };

    struct AudioMetrics
    {
        // Performance metrics
        float processingLatency = 0.0f;
        float networkLatency = 0.0f;
        float totalLatency = 0.0f;
        float cpuUsage = 0.0f;
        float memoryUsage = 0.0f;
        
        // Quality metrics
        float signalToNoiseRatio = 0.0f;
        float totalHarmonicDistortion = 0.0f;
        float frequencyResponse = 0.0f;
        float dynamicRange = 0.0f;
        
        // Network metrics
        int packetsTransmitted = 0;
        int packetsReceived = 0;
        int packetsLost = 0;
        float packetLossRate = 0.0f;
        float jitter = 0.0f;
        int bytesTransmitted = 0;
        int bytesReceived = 0;
        
        // Audio activity metrics
        int activeSources = 0;
        int mutedSources = 0;
        float averageVadLevel = 0.0f;
        float peakLevel = 0.0f;
        float rmsLevel = 0.0f;
        
        AZ_TYPE_INFO(AudioMetrics, "{44444444-5555-6666-7777-888888888888}");
    };

    // Audio callback types
    using AudioDataCallback = AZStd::function<void(const AZStd::vector<float>&, const AZStd::string&)>;
    using VoiceActivityCallback = AZStd::function<void(const AZStd::string&, bool)>;
    using AudioErrorCallback = AZStd::function<void(const AZStd::string&, const AZStd::string&)>;

    class SpatialAudioManager
    {
    public:
        SpatialAudioManager();
        ~SpatialAudioManager();

        // Initialization and cleanup
        bool Initialize(const AudioSettings& settings);
        void Shutdown();
        bool IsInitialized() const;

        // Audio device management
        AZStd::vector<AZStd::string> GetAvailableInputDevices();
        AZStd::vector<AZStd::string> GetAvailableOutputDevices();
        bool SetInputDevice(const AZStd::string& deviceId);
        bool SetOutputDevice(const AZStd::string& deviceId);
        AZStd::string GetCurrentInputDevice() const;
        AZStd::string GetCurrentOutputDevice() const;

        // User management
        bool AddUser(const AZStd::string& userId, const UserAudioProfile& profile);
        bool RemoveUser(const AZStd::string& userId);
        bool UpdateUserProfile(const AZStd::string& userId, const UserAudioProfile& profile);
        UserAudioProfile* GetUserProfile(const AZStd::string& userId);
        AZStd::vector<UserAudioProfile> GetAllUserProfiles();

        // Spatial positioning
        bool SetUserPosition(const AZStd::string& userId, const AZ::Vector3& position);
        bool SetUserOrientation(const AZStd::string& userId, const AZ::Quaternion& orientation);
        bool SetUserVelocity(const AZStd::string& userId, const AZ::Vector3& velocity);
        AZ::Vector3 GetUserPosition(const AZStd::string& userId) const;
        AZ::Quaternion GetUserOrientation(const AZStd::string& userId) const;
        AZ::Vector3 GetUserVelocity(const AZStd::string& userId) const;

        // Listener management
        bool SetListenerPosition(const AZ::Vector3& position);
        bool SetListenerOrientation(const AZ::Quaternion& orientation);
        bool SetListenerVelocity(const AZ::Vector3& velocity);
        AZ::Vector3 GetListenerPosition() const;
        AZ::Quaternion GetListenerOrientation() const;
        AZ::Vector3 GetListenerVelocity() const;

        // Audio transmission
        bool StartTransmission(const AZStd::string& userId);
        bool StopTransmission(const AZStd::string& userId);
        bool IsTransmitting(const AZStd::string& userId) const;
        bool SendAudioData(const AZStd::string& userId, const AZStd::vector<float>& audioData);
        bool SendEncodedAudioData(const AZStd::string& userId, const AZStd::vector<AZ::u8>& encodedData);

        // Audio reception
        bool StartReception(const AZStd::string& userId);
        bool StopReception(const AZStd::string& userId);
        bool IsReceiving(const AZStd::string& userId) const;
        AZStd::vector<float> GetAudioData(const AZStd::string& userId);
        AZStd::vector<AZ::u8> GetEncodedAudioData(const AZStd::string& userId);

        // Voice activity detection
        bool IsVoiceActive(const AZStd::string& userId) const;
        float GetVoiceActivityLevel(const AZStd::string& userId) const;
        void SetVoiceActivityThreshold(float threshold);
        float GetVoiceActivityThreshold() const;

        // Audio effects and processing
        bool SetUserGain(const AZStd::string& userId, float gain);
        bool SetUserMute(const AZStd::string& userId, bool muted);
        bool SetUserLocalMute(const AZStd::string& userId, bool localMuted);
        bool IsUserMuted(const AZStd::string& userId) const;
        bool IsUserLocalMuted(const AZStd::string& userId) const;
        float GetUserGain(const AZStd::string& userId) const;

        // Spatial audio effects
        bool SetOcclusionLevel(const AZStd::string& userId, float level);
        bool SetObstructionLevel(const AZStd::string& userId, float level);
        bool SetReverbLevel(const AZStd::string& userId, float level);
        float GetOcclusionLevel(const AZStd::string& userId) const;
        float GetObstructionLevel(const AZStd::string& userId) const;
        float GetReverbLevel(const AZStd::string& userId) const;

        // Environment management
        bool SetAudioEnvironment(const AudioEnvironment& environment);
        AudioEnvironment* GetCurrentAudioEnvironment();
        bool LoadAudioEnvironment(const AZStd::string& environmentId);
        AZStd::vector<AudioEnvironment> GetAvailableEnvironments();

        // Audio settings
        bool UpdateAudioSettings(const AudioSettings& settings);
        AudioSettings GetCurrentAudioSettings() const;
        bool SetSpatialAudioModel(SpatialAudioModel model);
        SpatialAudioModel GetSpatialAudioModel() const;
        bool SetAudioQuality(int quality);
        int GetAudioQuality() const;

        // Codec management
        bool SetAudioCodec(AudioCodec codec);
        AudioCodec GetAudioCodec() const;
        AZStd::vector<AudioCodec> GetSupportedCodecs();
        bool IsCodecSupported(AudioCodec codec) const;

        // Network adaptation
        bool SetNetworkLatency(float latency);
        bool SetPacketLoss(float packetLoss);
        bool SetBandwidth(int bandwidth);
        void AdaptToNetworkConditions();
        bool EnableAdaptiveQuality(bool enabled);

        // Audio visualization
        AZStd::vector<float> GetAudioSpectrum(const AZStd::string& userId);
        AZStd::vector<float> GetAudioWaveform(const AZStd::string& userId);
        float GetAudioLevel(const AZStd::string& userId) const;
        float GetPeakLevel(const AZStd::string& userId) const;
        float GetRMSLevel(const AZStd::string& userId) const;

        // 3D audio visualization
        AZStd::vector<AZ::Vector3> GetSoundRays(const AZStd::string& userId);
        AZStd::vector<AZ::Vector3> GetReflectionPaths(const AZStd::string& userId);
        AZStd::vector<float> GetSpatialGain(const AZStd::string& userId);
        void SetAudioVisualizationEnabled(bool enabled);
        bool IsAudioVisualizationEnabled() const;

        // Performance monitoring
        AudioMetrics GetAudioMetrics() const;
        float GetProcessingLatency() const;
        float GetNetworkLatency() const;
        float GetTotalLatency() const;
        float GetCPUUsage() const;
        float GetMemoryUsage() const;

        // Callback management
        void SetAudioDataCallback(AudioDataCallback callback);
        void SetVoiceActivityCallback(VoiceActivityCallback callback);
        void SetAudioErrorCallback(AudioErrorCallback callback);

        // Recording and playback
        bool StartRecording(const AZStd::string& filePath);
        bool StopRecording();
        bool IsRecording() const;
        bool PlayAudioFile(const AZStd::string& filePath, const AZ::Vector3& position);
        bool StopAudioFile(const AZStd::string& filePath);

        // Testing and calibration
        bool RunAudioTest();
        bool CalibrateAudio();
        bool TestEchoPath();
        bool TestSpatialAudio();
        float MeasureLatency();
        bool ValidateAudioSetup();

        // Advanced features
        bool EnableBinauralRecording(bool enabled);
        bool EnableAmbisonicRecording(bool enabled);
        bool SetHRTFProfile(const AZStd::string& profileId);
        bool LoadCustomHRTF(const AZStd::string& hrtfPath);
        bool EnableRoomCorrection(bool enabled);
        bool SetRoomCorrectionProfile(const AZStd::string& profileId);

        // Accessibility features
        bool EnableAudioDescription(bool enabled);
        bool SetAudioDescriptionLanguage(const AZStd::string& language);
        bool EnableHapticFeedback(bool enabled);
        bool SetHapticIntensity(float intensity);
        bool EnableVisualAudioIndicators(bool enabled);

        // Debug and diagnostics
        void EnableDebugMode(bool enabled);
        void SetLogLevel(int level);
        AZStd::string GetDebugInfo() const;
        void ExportAudioData(const AZStd::string& filePath);
        void ImportAudioData(const AZStd::string& filePath);

        // Update and processing
        void Update(float deltaTime);
        void ProcessAudio();
        void ProcessSpatialAudio();
        void ProcessNetworkAudio();

    private:
        // Internal initialization
        bool InitializeAudioSystem();
        bool InitializeAudioDevices();
        bool InitializeCodecs();
        bool InitializeEffects();
        bool InitializeSpatialProcessing();
        bool InitializeNetworking();

        // Audio processing
        void ProcessInputAudio();
        void ProcessOutputAudio();
        void ProcessSpatialTransform(const AZStd::string& userId, AZStd::vector<float>& audioData);
        void ApplyHRTF(const AZStd::string& userId, AZStd::vector<float>& audioData);
        void ApplyReverb(const AZStd::string& userId, AZStd::vector<float>& audioData);
        void ApplyOcclusion(const AZStd::string& userId, AZStd::vector<float>& audioData);
        void ApplyDistanceAttenuation(const AZStd::string& userId, AZStd::vector<float>& audioData);
        void ApplyDopplerEffect(const AZStd::string& userId, AZStd::vector<float>& audioData);

        // Codec processing
        AZStd::vector<AZ::u8> EncodeAudioData(const AZStd::vector<float>& audioData, AudioCodec codec);
        AZStd::vector<float> DecodeAudioData(const AZStd::vector<AZ::u8>& encodedData, AudioCodec codec);
        bool InitializeCodec(AudioCodec codec);
        void CleanupCodec(AudioCodec codec);

        // Network processing
        void ProcessNetworkPackets();
        void SendAudioPacket(const AZStd::string& userId, const AZStd::vector<AZ::u8>& data);
        void ReceiveAudioPacket(const AZStd::string& userId, const AZStd::vector<AZ::u8>& data);
        void HandlePacketLoss(const AZStd::string& userId);
        void HandleJitter(const AZStd::string& userId);

        // Audio effects processing
        void ProcessVoiceActivityDetection();
        void ProcessNoiseSuppression();
        void ProcessEchoCancellation();
        void ProcessAutomaticGainControl();
        void ProcessDynamicRangeCompression();

        // Spatial calculations
        float CalculateDistance(const AZ::Vector3& pos1, const AZ::Vector3& pos2) const;
        AZ::Vector3 CalculateDirection(const AZ::Vector3& from, const AZ::Vector3& to) const;
        float CalculateAttenuation(float distance) const;
        float CalculateOcclusion(const AZ::Vector3& from, const AZ::Vector3& to) const;
        AZStd::vector<AZ::Vector3> CalculateReflectionPaths(const AZ::Vector3& from, const AZ::Vector3& to) const;

        // Performance optimization
        void OptimizeAudioProcessing();
        void OptimizeNetworkTraffic();
        void OptimizeMemoryUsage();
        void UpdatePerformanceMetrics();

        // Error handling
        void HandleAudioError(const AZStd::string& error);
        void HandleNetworkError(const AZStd::string& error);
        void HandleCodecError(const AZStd::string& error);
        void RecoverFromError();

        // Member variables
        AZStd::unique_ptr<AudioSource> m_audioSource;
        AZStd::unique_ptr<AudioListener> m_audioListener;
        AZStd::unique_ptr<AudioProcessor> m_audioProcessor;
        AZStd::unique_ptr<AudioEffectsProcessor> m_effectsProcessor;
        AZStd::unique_ptr<VoiceActivityDetector> m_vadDetector;
        AZStd::unique_ptr<NoiseSuppressionFilter> m_noiseSuppressionFilter;
        AZStd::unique_ptr<EchoCancellationFilter> m_echoCancellationFilter;

        // Audio settings and state
        AudioSettings m_currentSettings;
        AZStd::unique_ptr<AudioEnvironment> m_currentEnvironment;
        AZStd::unordered_map<AZStd::string, UserAudioProfile> m_userProfiles;
        AZStd::unordered_map<AZStd::string, AZStd::unique_ptr<AudioSource>> m_userAudioSources;

        // Spatial audio state
        AZ::Vector3 m_listenerPosition;
        AZ::Quaternion m_listenerOrientation;
        AZ::Vector3 m_listenerVelocity;
        SpatialAudioModel m_spatialModel;
        float m_maxDistance;
        float m_referenceDistance;
        float m_rolloffFactor;
        float m_dopplerFactor;
        float m_speedOfSound;

        // Processing state
        bool m_isInitialized = false;
        bool m_isProcessing = false;
        bool m_isRecording = false;
        bool m_audioVisualizationEnabled = false;
        bool m_debugMode = false;
        int m_logLevel = 2;

        // Performance metrics
        AudioMetrics m_audioMetrics;
        float m_lastUpdateTime = 0.0f;
        float m_processingStartTime = 0.0f;

        // Callbacks
        AudioDataCallback m_audioDataCallback;
        VoiceActivityCallback m_voiceActivityCallback;
        AudioErrorCallback m_audioErrorCallback;

        // Threading and synchronization
        AZStd::mutex m_audioMutex;
        AZStd::mutex m_userProfilesMutex;
        AZStd::mutex m_metricsMutex;
        AZStd::mutex m_callbacksMutex;

        // Audio buffers
        AZStd::vector<float> m_inputBuffer;
        AZStd::vector<float> m_outputBuffer;
        AZStd::vector<float> m_processingBuffer;
        AZStd::unordered_map<AZStd::string, AZStd::vector<float>> m_userAudioBuffers;

        // Network buffers
        AZStd::unordered_map<AZStd::string, AZStd::vector<AZ::u8>> m_networkInputBuffers;
        AZStd::unordered_map<AZStd::string, AZStd::vector<AZ::u8>> m_networkOutputBuffers;

        // Device information
        AZStd::string m_currentInputDevice;
        AZStd::string m_currentOutputDevice;
        AZStd::vector<AZStd::string> m_availableInputDevices;
        AZStd::vector<AZStd::string> m_availableOutputDevices;

        // Codec information
        AudioCodec m_currentCodec;
        AZStd::vector<AudioCodec> m_supportedCodecs;
        AZStd::unordered_map<AudioCodec, void*> m_codecInstances;

        // Audio activity tracking
        float m_vadThreshold = 0.5f;
        AZStd::unordered_map<AZStd::string, float> m_userVadLevels;
        AZStd::unordered_map<AZStd::string, bool> m_userVoiceActivity;
        AZStd::unordered_map<AZStd::string, float> m_userLastActivity;
    };

    // Utility functions
    AZStd::string AudioCodecToString(AudioCodec codec);
    AudioCodec StringToAudioCodec(const AZStd::string& codecString);
    AZStd::string SpatialAudioModelToString(SpatialAudioModel model);
    SpatialAudioModel StringToSpatialAudioModel(const AZStd::string& modelString);
    AZStd::string ReverbPresetToString(ReverbPreset preset);
    ReverbPreset StringToReverbPreset(const AZStd::string& presetString);
    float DecibelToLinear(float decibel);
    float LinearToDecibel(float linear);
    float FrequencyToMel(float frequency);
    float MelToFrequency(float mel);

} // namespace Q3D 