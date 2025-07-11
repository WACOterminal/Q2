#pragma once

#include <AzCore/std/containers/unordered_map.h>
#include <AzCore/std/containers/vector.h>
#include <AzCore/std/containers/deque.h>
#include <AzCore/std/string/string.h>
#include <AzCore/std/smart_ptr/unique_ptr.h>
#include <AzCore/std/functional.h>
#include <AzCore/std/chrono/chrono.h>
#include <AzCore/std/parallel/mutex.h>
#include <AzCore/Component/TickBus.h>
#include <AzCore/Memory/SystemAllocator.h>

#include "WorkflowVisualization3DBus.h"

namespace Q3D
{
    // Forward declarations
    class FrameTimeAnalyzer;
    class MemoryProfiler;
    class GPUProfiler;
    class NetworkProfiler;
    class RenderingProfiler;
    class CPUProfiler;

    enum class PerformanceMetricType : int
    {
        FrameTime = 0,
        RenderTime = 1,
        UpdateTime = 2,
        CPUUsage = 3,
        GPUUsage = 4,
        MemoryUsage = 5,
        GPUMemoryUsage = 6,
        NetworkLatency = 7,
        DrawCalls = 8,
        Triangles = 9,
        Vertices = 10,
        TextureMemory = 11,
        AudioLatency = 12,
        LoadingTime = 13,
        CustomMetric = 14
    };

    enum class PerformanceLevel : int
    {
        Excellent = 0,
        Good = 1,
        Fair = 2,
        Poor = 3,
        Critical = 4
    };

    enum class OptimizationStrategy : int
    {
        Quality = 0,
        Performance = 1,
        Balanced = 2,
        PowerSaving = 3,
        Custom = 4
    };

    struct PerformanceThreshold
    {
        PerformanceMetricType metricType;
        float warningThreshold;
        float criticalThreshold;
        float targetValue;
        bool enableAutoOptimization = true;
        AZStd::string actionOnExceed;
        
        AZ_TYPE_INFO(PerformanceThreshold, "{11111111-2222-3333-4444-555555555555}");
    };

    struct PerformanceSample
    {
        PerformanceMetricType metricType;
        float value;
        float timestamp;
        AZStd::string context;
        AZStd::unordered_map<AZStd::string, float> additionalData;
        
        AZ_TYPE_INFO(PerformanceSample, "{22222222-3333-4444-5555-666666666666}");
    };

    struct PerformanceAlert
    {
        AZStd::string alertId;
        PerformanceMetricType metricType;
        PerformanceLevel severity;
        float value;
        float threshold;
        AZStd::string message;
        AZStd::string recommendation;
        float timestamp;
        bool isActive = true;
        bool hasBeenAcknowledged = false;
        
        AZ_TYPE_INFO(PerformanceAlert, "{33333333-4444-5555-6666-777777777777}");
    };

    struct OptimizationAction
    {
        AZStd::string actionId;
        AZStd::string actionType; // "reduce_quality", "limit_nodes", "disable_effects", etc.
        AZStd::string description;
        AZStd::unordered_map<AZStd::string, float> parameters;
        float expectedImprovement;
        float implementationCost;
        bool isReversible = true;
        bool isApplied = false;
        float appliedAt = 0.0f;
        
        AZ_TYPE_INFO(OptimizationAction, "{44444444-5555-6666-7777-888888888888}");
    };

    struct PerformanceProfile
    {
        AZStd::string profileId;
        AZStd::string name;
        AZStd::string description;
        QualityLevel qualityLevel;
        OptimizationStrategy strategy;
        
        // Quality settings
        bool enableShadows = true;
        bool enableReflections = true;
        bool enableParticles = true;
        bool enableAnimations = true;
        bool enableAA = true;
        int shadowResolution = 1024;
        float lodBias = 1.0f;
        int maxVisibleNodes = 1000;
        
        // Performance targets
        float targetFrameRate = 60.0f;
        float maxFrameTime = 16.67f; // 60 FPS
        float maxCPUUsage = 80.0f;
        float maxGPUUsage = 85.0f;
        float maxMemoryUsage = 512.0f; // MB
        
        // Thresholds
        AZStd::vector<PerformanceThreshold> thresholds;
        
        AZ_TYPE_INFO(PerformanceProfile, "{55555555-6666-7777-8888-999999999999}");
    };

    struct SystemCapabilities
    {
        // CPU information
        AZStd::string cpuName;
        int cpuCoreCount = 0;
        int cpuThreadCount = 0;
        float cpuFrequency = 0.0f;
        
        // GPU information
        AZStd::string gpuName;
        AZStd::string gpuVendor;
        int gpuMemoryMB = 0;
        int gpuComputeUnits = 0;
        
        // Memory information
        int systemMemoryMB = 0;
        int availableMemoryMB = 0;
        
        // Display information
        int screenWidth = 1920;
        int screenHeight = 1080;
        float screenDPI = 96.0f;
        bool supportsVR = false;
        bool supportsAR = false;
        
        // Feature support
        bool supportsCompute = false;
        bool supportsRayTracing = false;
        bool supportsMeshShaders = false;
        bool supportsVariableRateShading = false;
        
        // Performance estimates
        float estimatedPerformanceScore = 0.0f;
        PerformanceLevel estimatedPerformanceLevel = PerformanceLevel::Good;
        
        AZ_TYPE_INFO(SystemCapabilities, "{66666666-7777-8888-9999-AAAAAAAAAAAA}");
    };

    // Callback types
    using PerformanceAlertCallback = AZStd::function<void(const PerformanceAlert&)>;
    using OptimizationCallback = AZStd::function<void(const OptimizationAction&)>;
    using MetricsUpdateCallback = AZStd::function<void(const PerformanceMetrics&)>;

    class PerformanceMonitor
    {
    public:
        PerformanceMonitor();
        ~PerformanceMonitor();

        // Initialization and lifecycle
        bool Initialize();
        void Shutdown();
        bool IsInitialized() const;
        void Update(float deltaTime);

        // Metrics collection
        void RecordMetric(PerformanceMetricType type, float value, const AZStd::string& context = "");
        void RecordFrameTime(float frameTime);
        void RecordRenderTime(float renderTime);
        void RecordUpdateTime(float updateTime);
        void RecordMemoryUsage(float memoryMB);
        void RecordGPUUsage(float gpuUsage);
        void RecordCPUUsage(float cpuUsage);
        void RecordNetworkLatency(float latency);
        void RecordDrawCalls(int drawCalls);
        void RecordGeometry(int vertices, int triangles);

        // Metrics retrieval
        PerformanceMetrics GetCurrentMetrics() const;
        PerformanceMetrics GetAverageMetrics(float timeWindowSeconds = 5.0f) const;
        AZStd::vector<PerformanceSample> GetMetricHistory(PerformanceMetricType type, float timeWindowSeconds = 60.0f) const;
        float GetMetricValue(PerformanceMetricType type) const;
        float GetAverageMetricValue(PerformanceMetricType type, float timeWindowSeconds = 5.0f) const;
        float GetPeakMetricValue(PerformanceMetricType type, float timeWindowSeconds = 60.0f) const;

        // Performance analysis
        PerformanceLevel AnalyzePerformance() const;
        PerformanceLevel AnalyzeMetricPerformance(PerformanceMetricType type) const;
        AZStd::vector<PerformanceAlert> GetActiveAlerts() const;
        AZStd::vector<PerformanceAlert> GetRecentAlerts(float timeWindowSeconds = 300.0f) const;
        bool HasCriticalIssues() const;
        bool IsPerformanceAcceptable() const;

        // Automatic optimization
        void EnableAutoOptimization(bool enabled);
        bool IsAutoOptimizationEnabled() const;
        void SetOptimizationStrategy(OptimizationStrategy strategy);
        OptimizationStrategy GetOptimizationStrategy() const;
        AZStd::vector<OptimizationAction> GetSuggestedOptimizations() const;
        bool ApplyOptimization(const AZStd::string& actionId);
        bool RevertOptimization(const AZStd::string& actionId);

        // Performance profiles
        bool LoadPerformanceProfile(const AZStd::string& profileId);
        bool SavePerformanceProfile(const PerformanceProfile& profile);
        PerformanceProfile* GetCurrentProfile();
        AZStd::vector<PerformanceProfile> GetAvailableProfiles() const;
        PerformanceProfile CreateProfileForDevice(const SystemCapabilities& capabilities);
        void OptimizeCurrentProfile();

        // System capabilities
        SystemCapabilities DetectSystemCapabilities();
        SystemCapabilities GetSystemCapabilities() const;
        bool IsCapabilitySupported(const AZStd::string& capability) const;
        PerformanceLevel EstimatePerformanceLevel() const;
        QualityLevel RecommendQualityLevel() const;

        // Thresholds and alerts
        void SetPerformanceThreshold(const PerformanceThreshold& threshold);
        PerformanceThreshold* GetPerformanceThreshold(PerformanceMetricType type);
        void ClearPerformanceThreshold(PerformanceMetricType type);
        void AcknowledgeAlert(const AZStd::string& alertId);
        void ClearAlert(const AZStd::string& alertId);
        void ClearAllAlerts();

        // Quality settings management
        void ApplyQualitySettings(QualityLevel level);
        QualityLevel GetCurrentQualityLevel() const;
        void SetShadowsEnabled(bool enabled);
        void SetReflectionsEnabled(bool enabled);
        void SetParticlesEnabled(bool enabled);
        void SetAnimationsEnabled(bool enabled);
        void SetAntiAliasingEnabled(bool enabled);
        void SetShadowResolution(int resolution);
        void SetLODBias(float bias);
        void SetMaxVisibleNodes(int maxNodes);

        // Performance budgets
        void SetFrameTimeBudget(float budgetMS);
        void SetMemoryBudget(float budgetMB);
        void SetDrawCallBudget(int budget);
        void SetTriangleBudget(int budget);
        float GetFrameTimeBudget() const;
        float GetMemoryBudget() const;
        int GetDrawCallBudget() const;
        int GetTriangleBudget() const;
        bool IsWithinBudget(PerformanceMetricType type) const;

        // Real-time monitoring
        void StartProfiling();
        void StopProfiling();
        bool IsProfiling() const;
        void EnableContinuousMonitoring(bool enabled);
        bool IsContinuousMonitoringEnabled() const;
        void SetMonitoringFrequency(float frequencyHz);
        float GetMonitoringFrequency() const;

        // Detailed profiling
        void BeginProfilingSection(const AZStd::string& sectionName);
        void EndProfilingSection(const AZStd::string& sectionName);
        AZStd::unordered_map<AZStd::string, float> GetProfilingSectionTimes() const;
        void ResetProfilingData();

        // GPU profiling
        void BeginGPUProfiling();
        void EndGPUProfiling();
        float GetGPUFrameTime() const;
        AZStd::unordered_map<AZStd::string, float> GetGPUTimings() const;

        // Memory profiling
        void TakeMemorySnapshot();
        AZStd::unordered_map<AZStd::string, int> GetMemoryAllocationsByType() const;
        int GetTotalAllocatedMemory() const;
        int GetPeakMemoryUsage() const;
        AZStd::vector<AZStd::string> GetMemoryLeaks() const;

        // Network profiling
        void RecordNetworkPacket(int sizeBytes, bool incoming);
        float GetNetworkBandwidthUsage() const;
        int GetPacketsPerSecond() const;
        float GetPacketLossRate() const;

        // Performance reporting
        AZStd::string GeneratePerformanceReport() const;
        AZStd::string GenerateDetailedReport() const;
        void ExportPerformanceData(const AZStd::string& filePath) const;
        void ImportPerformanceData(const AZStd::string& filePath);

        // Callbacks
        void SetAlertCallback(PerformanceAlertCallback callback);
        void SetOptimizationCallback(OptimizationCallback callback);
        void SetMetricsUpdateCallback(MetricsUpdateCallback callback);

        // Configuration
        void SetSampleRetentionTime(float timeSeconds);
        void SetAlertCooldownTime(float timeSeconds);
        void EnableVerboseLogging(bool enabled);
        void SetLogLevel(int level);

        // Debug and testing
        void EnableDebugMode(bool enabled);
        void SimulatePerformanceIssue(PerformanceMetricType type, float severity);
        void InjectTestMetrics(const AZStd::vector<PerformanceSample>& samples);
        void ResetAllMetrics();

        // Statistics
        int GetTotalSampleCount() const;
        int GetAlertCount() const;
        int GetOptimizationCount() const;
        float GetUptimeSeconds() const;
        AZStd::unordered_map<AZStd::string, int> GetMetricCounts() const;

    private:
        // Internal monitoring
        void UpdateMetricsInternal();
        void ProcessPerformanceData();
        void CheckThresholds();
        void GenerateAlerts();
        void UpdateOptimizations();
        void CleanupOldData();

        // Analysis helpers
        PerformanceLevel CalculatePerformanceLevel(float value, const PerformanceThreshold& threshold) const;
        bool ShouldTriggerAlert(PerformanceMetricType type, float value) const;
        float CalculateTrend(PerformanceMetricType type, float timeWindowSeconds) const;
        bool IsMetricStable(PerformanceMetricType type, float timeWindowSeconds) const;

        // Optimization helpers
        AZStd::vector<OptimizationAction> GenerateOptimizationSuggestions() const;
        OptimizationAction CreateQualityReductionAction(float targetImprovement) const;
        OptimizationAction CreateNodeLimitAction(float targetImprovement) const;
        OptimizationAction CreateEffectDisableAction(const AZStd::string& effectType) const;
        void ApplyQualityOptimization(const OptimizationAction& action);
        void RevertQualityOptimization(const OptimizationAction& action);

        // System detection
        SystemCapabilities DetectCPUCapabilities() const;
        SystemCapabilities DetectGPUCapabilities() const;
        SystemCapabilities DetectMemoryCapabilities() const;
        SystemCapabilities DetectDisplayCapabilities() const;
        float CalculatePerformanceScore(const SystemCapabilities& capabilities) const;

        // Data management
        void AddSample(const PerformanceSample& sample);
        void RemoveOldSamples(float retentionTimeSeconds);
        AZStd::vector<PerformanceSample> GetSamplesInTimeWindow(PerformanceMetricType type, float timeWindowSeconds) const;
        float CalculateAverage(const AZStd::vector<PerformanceSample>& samples) const;
        float CalculatePeak(const AZStd::vector<PerformanceSample>& samples) const;

        // Alert management
        void CreateAlert(PerformanceMetricType type, PerformanceLevel severity, float value, float threshold);
        void UpdateAlert(const AZStd::string& alertId, bool isActive);
        void ExpireOldAlerts();

        // Member variables
        AZStd::unique_ptr<FrameTimeAnalyzer> m_frameTimeAnalyzer;
        AZStd::unique_ptr<MemoryProfiler> m_memoryProfiler;
        AZStd::unique_ptr<GPUProfiler> m_gpuProfiler;
        AZStd::unique_ptr<NetworkProfiler> m_networkProfiler;
        AZStd::unique_ptr<RenderingProfiler> m_renderingProfiler;
        AZStd::unique_ptr<CPUProfiler> m_cpuProfiler;

        // State
        bool m_isInitialized = false;
        bool m_isProfiling = false;
        bool m_continuousMonitoring = true;
        bool m_autoOptimizationEnabled = true;
        bool m_debugMode = false;
        bool m_verboseLogging = false;
        int m_logLevel = 2;

        // Configuration
        float m_monitoringFrequency = 10.0f; // Hz
        float m_sampleRetentionTime = 300.0f; // 5 minutes
        float m_alertCooldownTime = 30.0f; // 30 seconds
        OptimizationStrategy m_optimizationStrategy = OptimizationStrategy::Balanced;

        // Current metrics
        PerformanceMetrics m_currentMetrics;
        QualityLevel m_currentQualityLevel = QualityLevel::Medium;
        
        // Performance profile
        AZStd::unique_ptr<PerformanceProfile> m_currentProfile;
        AZStd::vector<PerformanceProfile> m_availableProfiles;
        
        // System capabilities
        SystemCapabilities m_systemCapabilities;
        
        // Data storage
        AZStd::unordered_map<PerformanceMetricType, AZStd::deque<PerformanceSample>> m_metricSamples;
        AZStd::unordered_map<PerformanceMetricType, PerformanceThreshold> m_thresholds;
        AZStd::unordered_map<AZStd::string, PerformanceAlert> m_alerts;
        AZStd::unordered_map<AZStd::string, OptimizationAction> m_optimizations;
        
        // Profiling data
        AZStd::unordered_map<AZStd::string, AZStd::chrono::steady_clock::time_point> m_profilingSectionStarts;
        AZStd::unordered_map<AZStd::string, float> m_profilingSectionTimes;
        AZStd::unordered_map<AZStd::string, float> m_gpuTimings;
        
        // Performance budgets
        float m_frameTimeBudget = 16.67f; // 60 FPS
        float m_memoryBudget = 512.0f; // MB
        int m_drawCallBudget = 1000;
        int m_triangleBudget = 100000;
        
        // Callbacks
        PerformanceAlertCallback m_alertCallback;
        OptimizationCallback m_optimizationCallback;
        MetricsUpdateCallback m_metricsUpdateCallback;
        
        // Timing
        AZStd::chrono::steady_clock::time_point m_startTime;
        AZStd::chrono::steady_clock::time_point m_lastUpdateTime;
        AZStd::chrono::steady_clock::time_point m_lastAlertCheckTime;
        
        // Statistics
        int m_totalSampleCount = 0;
        int m_alertCount = 0;
        int m_optimizationCount = 0;
        
        // Threading
        mutable AZStd::mutex m_metricsMutex;
        mutable AZStd::mutex m_alertsMutex;
        mutable AZStd::mutex m_optimizationsMutex;
        mutable AZStd::mutex m_profilingMutex;
    };

    // Utility functions
    AZStd::string PerformanceMetricTypeToString(PerformanceMetricType type);
    PerformanceMetricType StringToPerformanceMetricType(const AZStd::string& typeString);
    AZStd::string PerformanceLevelToString(PerformanceLevel level);
    PerformanceLevel StringToPerformanceLevel(const AZStd::string& levelString);
    AZStd::string OptimizationStrategyToString(OptimizationStrategy strategy);
    OptimizationStrategy StringToOptimizationStrategy(const AZStd::string& strategyString);
    
    // Performance calculation utilities
    float CalculateFramesPerSecond(float frameTimeMS);
    float CalculateFrameTimeMS(float framesPerSecond);
    float ConvertBytesToMB(int bytes);
    int ConvertMBToBytes(float megabytes);
    float CalculatePercentage(float value, float total);
    float NormalizeMetricValue(PerformanceMetricType type, float value);

} // namespace Q3D 