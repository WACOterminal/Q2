#pragma once

#include <AzCore/std/containers/vector.h>
#include <AzCore/std/containers/unordered_map.h>
#include <AzCore/std/string/string.h>
#include <AzCore/std/smart_ptr/unique_ptr.h>
#include <AzCore/std/functional.h>
#include <AzCore/Math/Vector3.h>
#include <AzCore/Math/Matrix3x3.h>
#include <AzCore/Math/Random.h>
#include <AzCore/std/parallel/mutex.h>
#include <AzCore/std/parallel/thread.h>
#include <AzCore/std/chrono/chrono.h>

#include "WorkflowData.h"
#include "WorkflowVisualization3DBus.h"

namespace Q3D
{
    // Forward declarations
    class NeuralNetwork;
    class GeneticAlgorithm;
    class SimulatedAnnealing;
    class ParticleSwarmOptimizer;
    class LayoutEvaluator;
    class ContextualAI;

    enum class OptimizationAlgorithm : int
    {
        ForceDirected = 0,
        Hierarchical = 1,
        Circular = 2,
        Grid = 3,
        NeuralNetwork = 4,
        GeneticAlgorithm = 5,
        SimulatedAnnealing = 6,
        ParticleSwarm = 7,
        HybridAI = 8,
        ContextualAI = 9,
        AdaptiveLearning = 10
    };

    enum class LayoutObjective : int
    {
        MinimizeOverlap = 0,
        MinimizeEdgeCrossings = 1,
        MaximizeAesthetics = 2,
        MaximizeReadability = 3,
        MinimizeArea = 4,
        MaximizeSymmetry = 5,
        MinimizeNodeDistance = 6,
        MaximizeFlowClarity = 7,
        OptimizePerformance = 8,
        EnhanceAccessibility = 9,
        SupportCollaboration = 10,
        MultiObjective = 11
    };

    enum class LayoutStrategy : int
    {
        Sequential = 0,
        Parallel = 1,
        Iterative = 2,
        Adaptive = 3,
        Hierarchical = 4,
        Collaborative = 5,
        RealTime = 6,
        Predictive = 7
    };

    struct LayoutConstraint
    {
        AZStd::string constraintId;
        AZStd::string type; // "position", "distance", "angle", "group", "layer", "accessibility"
        AZStd::vector<AZStd::string> nodeIds;
        AZStd::unordered_map<AZStd::string, float> parameters;
        float weight = 1.0f;
        bool isHard = false; // Hard constraints must be satisfied
        bool isActive = true;
        AZStd::string description;
        
        AZ_TYPE_INFO(LayoutConstraint, "{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE}");
    };

    struct LayoutPreference
    {
        AZStd::string preferenceId;
        AZStd::string userId;
        AZStd::string type; // "style", "spacing", "orientation", "grouping", "color", "size"
        AZStd::unordered_map<AZStd::string, float> values;
        float importance = 1.0f;
        bool isLearned = false;
        AZStd::string context;
        
        AZ_TYPE_INFO(LayoutPreference, "{BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF}");
    };

    struct LayoutMetrics
    {
        // Aesthetic metrics
        float aestheticScore = 0.0f;
        float symmetryScore = 0.0f;
        float balanceScore = 0.0f;
        float harmonyScore = 0.0f;
        
        // Readability metrics
        float readabilityScore = 0.0f;
        float clarityScore = 0.0f;
        float organizationScore = 0.0f;
        float flowScore = 0.0f;
        
        // Efficiency metrics
        float compactnessScore = 0.0f;
        float edgeCrossingScore = 0.0f;
        float overlapScore = 0.0f;
        float distanceScore = 0.0f;
        
        // Performance metrics
        float renderingScore = 0.0f;
        float interactionScore = 0.0f;
        float navigationScore = 0.0f;
        float memoryScore = 0.0f;
        
        // Collaboration metrics
        float collaborationScore = 0.0f;
        float accessibilityScore = 0.0f;
        float shareabilityScore = 0.0f;
        float adaptabilityScore = 0.0f;
        
        // Overall score
        float overallScore = 0.0f;
        float confidence = 0.0f;
        
        AZ_TYPE_INFO(LayoutMetrics, "{CCCCCCCC-DDDD-EEEE-FFFF-000000000000}");
    };

    struct OptimizationParameters
    {
        OptimizationAlgorithm algorithm = OptimizationAlgorithm::HybridAI;
        LayoutObjective objective = LayoutObjective::MultiObjective;
        LayoutStrategy strategy = LayoutStrategy::Adaptive;
        
        // Algorithm-specific parameters
        int maxIterations = 1000;
        float convergenceThreshold = 0.001f;
        float learningRate = 0.1f;
        float mutationRate = 0.1f;
        float crossoverRate = 0.8f;
        int populationSize = 50;
        float temperature = 1000.0f;
        float coolingRate = 0.95f;
        
        // Multi-objective weights
        AZStd::unordered_map<AZStd::string, float> objectiveWeights;
        
        // Constraints and preferences
        AZStd::vector<LayoutConstraint> constraints;
        AZStd::vector<LayoutPreference> preferences;
        
        // Context information
        AZStd::string workflowType;
        AZStd::string userExperience;
        AZStd::string deviceType;
        AZStd::string collaborationContext;
        bool isRealTime = false;
        
        // Performance constraints
        float maxOptimizationTime = 30.0f; // seconds
        float targetFrameRate = 60.0f;
        int maxMemoryUsage = 512; // MB
        
        AZ_TYPE_INFO(OptimizationParameters, "{DDDDDDDD-EEEE-FFFF-0000-111111111111}");
    };

    struct LayoutSolution
    {
        AZStd::string solutionId;
        AZStd::unordered_map<AZStd::string, AZ::Vector3> nodePositions;
        AZStd::unordered_map<AZStd::string, AZ::Vector3> nodeOrientations;
        AZStd::unordered_map<AZStd::string, float> nodeScales;
        AZStd::unordered_map<AZStd::string, AZStd::vector<AZ::Vector3>> edgePaths;
        
        // Camera positioning
        AZ::Vector3 cameraPosition;
        AZ::Vector3 cameraTarget;
        float cameraFOV = 60.0f;
        
        // Evaluation metrics
        LayoutMetrics metrics;
        float fitness = 0.0f;
        float confidence = 0.0f;
        
        // Optimization metadata
        OptimizationAlgorithm algorithm;
        int iterations = 0;
        float optimizationTime = 0.0f;
        AZStd::string optimizationContext;
        
        // Validation
        bool isValid = true;
        AZStd::vector<AZStd::string> validationErrors;
        
        AZ_TYPE_INFO(LayoutSolution, "{EEEEEEEE-FFFF-0000-1111-222222222222}");
    };

    struct LearningData
    {
        AZStd::string dataId;
        AZStd::string userId;
        AZStd::string workflowType;
        WorkflowData originalWorkflow;
        LayoutSolution chosenSolution;
        AZStd::vector<LayoutSolution> alternativeSolutions;
        
        // User feedback
        float userRating = 0.0f;
        AZStd::string userFeedback;
        AZStd::unordered_map<AZStd::string, float> featureRatings;
        
        // Usage statistics
        float usageTime = 0.0f;
        int interactionCount = 0;
        int modificationCount = 0;
        float taskCompletionTime = 0.0f;
        
        // Context
        AZStd::string deviceType;
        AZStd::string collaborationContext;
        AZStd::string timestamp;
        
        AZ_TYPE_INFO(LearningData, "{FFFFFFFF-0000-1111-2222-333333333333}");
    };

    // Callback types
    using OptimizationProgressCallback = AZStd::function<void(float, const AZStd::string&)>;
    using OptimizationCompleteCallback = AZStd::function<void(const LayoutSolution&)>;
    using LearningUpdateCallback = AZStd::function<void(const LearningData&)>;

    class AILayoutOptimizer
    {
    public:
        AILayoutOptimizer();
        ~AILayoutOptimizer();

        // Initialization and setup
        bool Initialize();
        void Shutdown();
        bool IsInitialized() const;

        // Main optimization interface
        LayoutSolution OptimizeLayout(const WorkflowData& workflow, const OptimizationParameters& params);
        AZStd::vector<LayoutSolution> GenerateAlternatives(const WorkflowData& workflow, const OptimizationParameters& params, int numAlternatives = 3);
        bool OptimizeLayoutAsync(const WorkflowData& workflow, const OptimizationParameters& params, OptimizationCompleteCallback callback);
        void CancelOptimization();

        // Real-time optimization
        bool StartRealTimeOptimization(const WorkflowData& workflow, const OptimizationParameters& params);
        void StopRealTimeOptimization();
        bool IsRealTimeOptimizationActive() const;
        LayoutSolution GetCurrentOptimization() const;

        // Algorithm-specific optimization
        LayoutSolution OptimizeWithForceDirected(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution OptimizeWithHierarchical(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution OptimizeWithCircular(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution OptimizeWithGrid(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution OptimizeWithNeuralNetwork(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution OptimizeWithGeneticAlgorithm(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution OptimizeWithSimulatedAnnealing(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution OptimizeWithParticleSwarm(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution OptimizeWithHybridAI(const WorkflowData& workflow, const OptimizationParameters& params);

        // Contextual and adaptive optimization
        LayoutSolution OptimizeWithContext(const WorkflowData& workflow, const OptimizationParameters& params, const AZStd::string& context);
        LayoutSolution OptimizeForCollaboration(const WorkflowData& workflow, const OptimizationParameters& params, const AZStd::vector<AZStd::string>& userIds);
        LayoutSolution OptimizeForAccessibility(const WorkflowData& workflow, const OptimizationParameters& params, const AZStd::vector<AZStd::string>& accessibilityRequirements);
        LayoutSolution OptimizeForDevice(const WorkflowData& workflow, const OptimizationParameters& params, const AZStd::string& deviceType);

        // Solution evaluation and comparison
        LayoutMetrics EvaluateSolution(const LayoutSolution& solution, const WorkflowData& workflow);
        float CompareSolutions(const LayoutSolution& solution1, const LayoutSolution& solution2);
        LayoutSolution SelectBestSolution(const AZStd::vector<LayoutSolution>& solutions);
        AZStd::vector<LayoutSolution> RankSolutions(const AZStd::vector<LayoutSolution>& solutions);

        // Constraint and preference management
        bool AddConstraint(const LayoutConstraint& constraint);
        bool RemoveConstraint(const AZStd::string& constraintId);
        bool UpdateConstraint(const LayoutConstraint& constraint);
        LayoutConstraint* GetConstraint(const AZStd::string& constraintId);
        AZStd::vector<LayoutConstraint> GetAllConstraints();

        bool AddPreference(const LayoutPreference& preference);
        bool RemovePreference(const AZStd::string& preferenceId);
        bool UpdatePreference(const LayoutPreference& preference);
        LayoutPreference* GetPreference(const AZStd::string& preferenceId);
        AZStd::vector<LayoutPreference> GetUserPreferences(const AZStd::string& userId);

        // Machine learning and adaptation
        bool TrainModel(const AZStd::vector<LearningData>& trainingData);
        bool UpdateModel(const LearningData& newData);
        bool SaveModel(const AZStd::string& modelPath);
        bool LoadModel(const AZStd::string& modelPath);
        float GetModelAccuracy() const;
        AZStd::string GetModelInfo() const;

        // User feedback and learning
        void RecordUserFeedback(const AZStd::string& solutionId, float rating, const AZStd::string& feedback);
        void RecordUserInteraction(const AZStd::string& solutionId, const AZStd::string& interactionType, const AZStd::unordered_map<AZStd::string, float>& data);
        void RecordUsageStatistics(const AZStd::string& solutionId, float usageTime, int interactionCount);
        AZStd::vector<LearningData> GetLearningData(const AZStd::string& userId = "");

        // Personalization and adaptation
        void PersonalizeForUser(const AZStd::string& userId);
        void AdaptToWorkflowType(const AZStd::string& workflowType);
        void AdaptToCollaborationContext(const AZStd::string& collaborationContext);
        void AdaptToDevice(const AZStd::string& deviceType);

        // Performance optimization
        void SetPerformanceMode(bool enabled);
        void SetMaxOptimizationTime(float timeSeconds);
        void SetTargetFrameRate(float fps);
        void SetMemoryLimit(int megabytes);
        void EnableParallelProcessing(bool enabled);
        void SetThreadCount(int threadCount);

        // Validation and quality assurance
        bool ValidateSolution(const LayoutSolution& solution, const WorkflowData& workflow);
        AZStd::vector<AZStd::string> GetValidationErrors(const LayoutSolution& solution, const WorkflowData& workflow);
        bool RepairSolution(LayoutSolution& solution, const WorkflowData& workflow);
        LayoutSolution SanitizeSolution(const LayoutSolution& solution, const WorkflowData& workflow);

        // Debugging and analysis
        void EnableDebugMode(bool enabled);
        void SetLogLevel(int level);
        AZStd::string GetOptimizationReport(const AZStd::string& solutionId);
        AZStd::string GetPerformanceReport();
        void ExportOptimizationData(const AZStd::string& filePath);
        void ImportOptimizationData(const AZStd::string& filePath);

        // Callback management
        void SetProgressCallback(OptimizationProgressCallback callback);
        void SetCompleteCallback(OptimizationCompleteCallback callback);
        void SetLearningUpdateCallback(LearningUpdateCallback callback);

        // Configuration
        void SetDefaultAlgorithm(OptimizationAlgorithm algorithm);
        void SetDefaultObjective(LayoutObjective objective);
        void SetDefaultStrategy(LayoutStrategy strategy);
        OptimizationParameters GetDefaultParameters() const;
        void SetDefaultParameters(const OptimizationParameters& params);

        // Statistics and monitoring
        int GetOptimizationCount() const;
        float GetAverageOptimizationTime() const;
        float GetSuccessRate() const;
        AZStd::unordered_map<AZStd::string, int> GetAlgorithmUsageStats();
        AZStd::unordered_map<AZStd::string, float> GetPerformanceMetrics();

    private:
        // Algorithm implementations
        LayoutSolution RunForceDirectedOptimization(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution RunHierarchicalOptimization(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution RunCircularOptimization(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution RunGridOptimization(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution RunNeuralNetworkOptimization(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution RunGeneticAlgorithmOptimization(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution RunSimulatedAnnealingOptimization(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution RunParticleSwarmOptimization(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution RunHybridAIOptimization(const WorkflowData& workflow, const OptimizationParameters& params);

        // Evaluation functions
        float EvaluateAesthetics(const LayoutSolution& solution, const WorkflowData& workflow);
        float EvaluateReadability(const LayoutSolution& solution, const WorkflowData& workflow);
        float EvaluateEfficiency(const LayoutSolution& solution, const WorkflowData& workflow);
        float EvaluatePerformance(const LayoutSolution& solution, const WorkflowData& workflow);
        float EvaluateCollaboration(const LayoutSolution& solution, const WorkflowData& workflow);
        float EvaluateAccessibility(const LayoutSolution& solution, const WorkflowData& workflow);

        // Constraint checking
        bool CheckConstraints(const LayoutSolution& solution, const WorkflowData& workflow);
        bool CheckHardConstraints(const LayoutSolution& solution, const WorkflowData& workflow);
        float EvaluateSoftConstraints(const LayoutSolution& solution, const WorkflowData& workflow);
        bool RepairConstraintViolations(LayoutSolution& solution, const WorkflowData& workflow);

        // Preference integration
        float EvaluatePreferences(const LayoutSolution& solution, const WorkflowData& workflow);
        void ApplyUserPreferences(LayoutSolution& solution, const AZStd::string& userId);
        void LearnUserPreferences(const AZStd::string& userId, const LayoutSolution& solution, float rating);

        // Optimization utilities
        LayoutSolution CreateInitialSolution(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution CreateRandomSolution(const WorkflowData& workflow, const OptimizationParameters& params);
        LayoutSolution MutateSolution(const LayoutSolution& solution, float mutationRate);
        LayoutSolution CrossoverSolutions(const LayoutSolution& parent1, const LayoutSolution& parent2);
        LayoutSolution LocalSearch(const LayoutSolution& solution, const WorkflowData& workflow);

        // Machine learning components
        void InitializeNeuralNetwork();
        void TrainNeuralNetwork(const AZStd::vector<LearningData>& trainingData);
        LayoutSolution PredictOptimalLayout(const WorkflowData& workflow, const OptimizationParameters& params);
        void UpdateNeuralNetwork(const LearningData& newData);

        // Contextual AI
        void InitializeContextualAI();
        AZStd::string AnalyzeContext(const WorkflowData& workflow, const OptimizationParameters& params);
        OptimizationParameters AdaptParametersToContext(const OptimizationParameters& params, const AZStd::string& context);
        LayoutSolution ApplyContextualOptimization(const LayoutSolution& solution, const AZStd::string& context);

        // Performance monitoring
        void UpdatePerformanceMetrics();
        void OptimizePerformance();
        bool CheckPerformanceConstraints(const OptimizationParameters& params);

        // Threading and parallelization
        void InitializeThreadPool();
        void ShutdownThreadPool();
        void DistributeOptimizationTasks();
        void CollectOptimizationResults();

        // Data management
        void SaveOptimizationResult(const LayoutSolution& solution);
        void LoadOptimizationHistory();
        void CleanupOptimizationData();

        // Member variables
        AZStd::unique_ptr<NeuralNetwork> m_neuralNetwork;
        AZStd::unique_ptr<GeneticAlgorithm> m_geneticAlgorithm;
        AZStd::unique_ptr<SimulatedAnnealing> m_simulatedAnnealing;
        AZStd::unique_ptr<ParticleSwarmOptimizer> m_particleSwarmOptimizer;
        AZStd::unique_ptr<LayoutEvaluator> m_layoutEvaluator;
        AZStd::unique_ptr<ContextualAI> m_contextualAI;

        // State management
        bool m_isInitialized = false;
        bool m_isOptimizing = false;
        bool m_isRealTimeOptimizing = false;
        bool m_performanceMode = false;
        bool m_debugMode = false;
        int m_logLevel = 2;

        // Configuration
        OptimizationParameters m_defaultParameters;
        OptimizationAlgorithm m_defaultAlgorithm = OptimizationAlgorithm::HybridAI;
        LayoutObjective m_defaultObjective = LayoutObjective::MultiObjective;
        LayoutStrategy m_defaultStrategy = LayoutStrategy::Adaptive;

        // Constraints and preferences
        AZStd::unordered_map<AZStd::string, LayoutConstraint> m_constraints;
        AZStd::unordered_map<AZStd::string, LayoutPreference> m_preferences;
        AZStd::unordered_map<AZStd::string, AZStd::vector<LayoutPreference>> m_userPreferences;

        // Learning data
        AZStd::vector<LearningData> m_learningData;
        AZStd::unordered_map<AZStd::string, LayoutSolution> m_optimizationHistory;

        // Performance tracking
        int m_optimizationCount = 0;
        float m_totalOptimizationTime = 0.0f;
        int m_successfulOptimizations = 0;
        AZStd::unordered_map<AZStd::string, int> m_algorithmUsageStats;
        AZStd::unordered_map<AZStd::string, float> m_performanceMetrics;

        // Threading
        AZStd::vector<AZStd::thread> m_workerThreads;
        AZStd::mutex m_optimizationMutex;
        AZStd::mutex m_learningDataMutex;
        AZStd::mutex m_performanceMetricsMutex;
        bool m_parallelProcessingEnabled = true;
        int m_threadCount = 4;

        // Callbacks
        OptimizationProgressCallback m_progressCallback;
        OptimizationCompleteCallback m_completeCallback;
        LearningUpdateCallback m_learningUpdateCallback;

        // Performance constraints
        float m_maxOptimizationTime = 30.0f;
        float m_targetFrameRate = 60.0f;
        int m_memoryLimit = 512;

        // Random number generation
        AZ::SimpleLcgRandom m_random;
        
        // Current optimization context
        AZStd::string m_currentOptimizationId;
        AZStd::chrono::steady_clock::time_point m_optimizationStartTime;
        volatile bool m_shouldCancelOptimization = false;
    };

    // Utility functions
    AZStd::string OptimizationAlgorithmToString(OptimizationAlgorithm algorithm);
    OptimizationAlgorithm StringToOptimizationAlgorithm(const AZStd::string& algorithmString);
    AZStd::string LayoutObjectiveToString(LayoutObjective objective);
    LayoutObjective StringToLayoutObjective(const AZStd::string& objectiveString);
    AZStd::string LayoutStrategyToString(LayoutStrategy strategy);
    LayoutStrategy StringToLayoutStrategy(const AZStd::string& strategyString);
    
    // Mathematical utilities
    float CalculateEuclideanDistance(const AZ::Vector3& point1, const AZ::Vector3& point2);
    float CalculateManhattanDistance(const AZ::Vector3& point1, const AZ::Vector3& point2);
    AZ::Vector3 CalculateCentroid(const AZStd::vector<AZ::Vector3>& points);
    AZ::Matrix3x3 CalculateCovariance(const AZStd::vector<AZ::Vector3>& points);
    float CalculateVariance(const AZStd::vector<float>& values);
    float CalculateStandardDeviation(const AZStd::vector<float>& values);
    
    // Layout analysis utilities
    int CountEdgeCrossings(const AZStd::unordered_map<AZStd::string, AZ::Vector3>& nodePositions, const AZStd::vector<WorkflowEdge>& edges);
    float CalculateOverlapArea(const AZStd::unordered_map<AZStd::string, AZ::Vector3>& nodePositions, float nodeRadius = 1.0f);
    float CalculateLayoutSymmetry(const AZStd::unordered_map<AZStd::string, AZ::Vector3>& nodePositions);
    float CalculateLayoutBalance(const AZStd::unordered_map<AZStd::string, AZ::Vector3>& nodePositions);

} // namespace Q3D 