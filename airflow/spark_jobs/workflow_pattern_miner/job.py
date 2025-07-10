"""
Workflow Pattern Mining Spark Job

This job analyzes historical workflow execution data to identify:
1. Common execution patterns
2. Performance bottlenecks
3. Success/failure patterns
4. Optimal agent assignments
5. Workflow optimization opportunities
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, 
    DoubleType, ArrayType, MapType, BooleanType
)
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.stat import Correlation

import httpx
import pulsar

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IGNITE_ADDRESSES = os.getenv("IGNITE_ADDRESSES", "ignite:10800").split(",")
PULSAR_URL = os.getenv("PULSAR_URL", "pulsar://pulsar:6650")
KNOWLEDGE_GRAPH_URL = os.getenv("KNOWLEDGE_GRAPH_URL", "http://knowledgegraphq:8006")
PLATFORM_EVENTS_TOPIC = "persistent://public/default/q.platform.events"

# Pattern mining parameters
MIN_SUPPORT = 0.01  # Minimum support for frequent patterns
MIN_CONFIDENCE = 0.5  # Minimum confidence for association rules
SEQUENCE_LENGTH = 5  # Maximum sequence length to analyze
PERFORMANCE_PERCENTILE = 90  # Percentile for bottleneck detection


class WorkflowPatternMiner:
    """Analyzes workflow execution patterns to identify optimization opportunities"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.pulsar_client = None
        self.producer = None
        
    def initialize_pulsar(self):
        """Initialize Pulsar client for publishing results"""
        self.pulsar_client = pulsar.Client(PULSAR_URL)
        self.producer = self.pulsar_client.create_producer(PLATFORM_EVENTS_TOPIC)
        
    def load_workflow_data(self) -> DataFrame:
        """Load historical workflow execution data from data store"""
        # In production, this would load from Cassandra/JanusGraph
        # For now, we'll create sample schema
        
        schema = StructType([
            StructField("workflow_id", StringType(), False),
            StructField("workflow_type", StringType(), False),
            StructField("status", StringType(), False),
            StructField("start_time", LongType(), False),
            StructField("end_time", LongType(), True),
            StructField("duration_ms", LongType(), True),
            StructField("tasks", ArrayType(StructType([
                StructField("task_id", StringType(), False),
                StructField("task_type", StringType(), False),
                StructField("agent_personality", StringType(), False),
                StructField("status", StringType(), False),
                StructField("start_time", LongType(), False),
                StructField("end_time", LongType(), True),
                StructField("duration_ms", LongType(), True),
                StructField("retry_count", LongType(), True),
                StructField("dependencies", ArrayType(StringType()), True)
            ])), True),
            StructField("metadata", MapType(StringType(), StringType()), True)
        ])
        
        # Load data - in production this would be from actual storage
        logger.info("Loading workflow execution data...")
        
        # For demonstration, generate sample data
        # In production, replace with actual data loading:
        # df = spark.read.format("org.apache.spark.sql.cassandra") \
        #     .options(table="workflow_executions", keyspace="qagi") \
        #     .load()
        
        df = self.spark.createDataFrame([], schema)
        return df
        
    def extract_sequence_patterns(self, df: DataFrame) -> DataFrame:
        """Extract common task execution sequences"""
        logger.info("Extracting sequence patterns...")
        
        # Explode tasks and create sequences
        task_sequences = df.select(
            "workflow_id",
            "workflow_type",
            "status",
            F.explode("tasks").alias("task")
        ).select(
            "workflow_id",
            "workflow_type",
            "status",
            F.col("task.task_type").alias("task_type"),
            F.col("task.agent_personality").alias("agent"),
            F.col("task.start_time").alias("start_time")
        )
        
        # Create task sequences using window functions
        window_spec = Window.partitionBy("workflow_id").orderBy("start_time")
        
        sequences = task_sequences.withColumn(
            "task_sequence",
            F.collect_list("task_type").over(
                window_spec.rowsBetween(0, SEQUENCE_LENGTH - 1)
            )
        ).filter(
            F.size("task_sequence") >= 2
        )
        
        # Count sequence frequencies
        sequence_counts = sequences.groupBy("task_sequence").agg(
            F.count("*").alias("count"),
            F.avg(F.when(F.col("status") == "SUCCESS", 1.0).otherwise(0.0)).alias("success_rate")
        ).filter(
            F.col("count") >= MIN_SUPPORT * sequences.count()
        )
        
        return sequence_counts
        
    def detect_bottlenecks(self, df: DataFrame) -> DataFrame:
        """Identify performance bottlenecks in workflows"""
        logger.info("Detecting performance bottlenecks...")
        
        # Calculate task-level statistics
        task_stats = df.select(
            F.explode("tasks").alias("task")
        ).select(
            F.col("task.task_type").alias("task_type"),
            F.col("task.agent_personality").alias("agent"),
            F.col("task.duration_ms").alias("duration"),
            F.col("task.retry_count").alias("retries")
        ).groupBy("task_type", "agent").agg(
            F.avg("duration").alias("avg_duration"),
            F.stddev("duration").alias("stddev_duration"),
            F.percentile_approx("duration", PERFORMANCE_PERCENTILE / 100.0).alias("p90_duration"),
            F.max("duration").alias("max_duration"),
            F.avg("retries").alias("avg_retries"),
            F.count("*").alias("execution_count")
        )
        
        # Identify bottlenecks (tasks with high p90 duration)
        overall_p90 = task_stats.agg(
            F.percentile_approx("p90_duration", 0.9).alias("overall_p90")
        ).collect()[0]["overall_p90"]
        
        bottlenecks = task_stats.filter(
            F.col("p90_duration") > overall_p90
        ).withColumn(
            "bottleneck_score",
            F.col("p90_duration") / overall_p90
        )
        
        return bottlenecks
        
    def analyze_success_factors(self, df: DataFrame) -> DataFrame:
        """Analyze factors contributing to workflow success"""
        logger.info("Analyzing success factors...")
        
        # Create feature vectors for success analysis
        workflow_features = df.select(
            "workflow_id",
            "workflow_type",
            F.when(F.col("status") == "SUCCESS", 1.0).otherwise(0.0).alias("success"),
            F.col("duration_ms"),
            F.size("tasks").alias("task_count"),
            F.expr("aggregate(tasks, 0L, (acc, x) -> acc + x.retry_count)").alias("total_retries")
        )
        
        # Calculate success correlations by workflow type
        success_factors = workflow_features.groupBy("workflow_type").agg(
            F.avg("success").alias("success_rate"),
            F.corr("task_count", "success").alias("task_count_correlation"),
            F.corr("duration_ms", "success").alias("duration_correlation"),
            F.corr("total_retries", "success").alias("retry_correlation"),
            F.count("*").alias("sample_size")
        )
        
        return success_factors
        
    def mine_agent_performance_patterns(self, df: DataFrame) -> DataFrame:
        """Analyze agent performance patterns"""
        logger.info("Mining agent performance patterns...")
        
        # Extract agent performance metrics
        agent_tasks = df.select(
            F.explode("tasks").alias("task"),
            "workflow_type"
        ).select(
            "workflow_type",
            F.col("task.task_type").alias("task_type"),
            F.col("task.agent_personality").alias("agent"),
            F.col("task.status").alias("status"),
            F.col("task.duration_ms").alias("duration")
        )
        
        # Calculate agent performance by task type
        agent_performance = agent_tasks.groupBy(
            "agent", "task_type", "workflow_type"
        ).agg(
            F.avg("duration").alias("avg_duration"),
            F.avg(F.when(F.col("status") == "SUCCESS", 1.0).otherwise(0.0)).alias("success_rate"),
            F.count("*").alias("execution_count")
        ).filter(
            F.col("execution_count") >= 10  # Minimum sample size
        )
        
        # Find best agent for each task type
        window_spec = Window.partitionBy("task_type", "workflow_type").orderBy(
            F.desc("success_rate"), F.asc("avg_duration")
        )
        
        best_agents = agent_performance.withColumn(
            "rank",
            F.row_number().over(window_spec)
        ).filter(
            F.col("rank") == 1
        )
        
        return best_agents
        
    def generate_optimization_recommendations(
        self, 
        sequences: DataFrame,
        bottlenecks: DataFrame,
        success_factors: DataFrame,
        agent_patterns: DataFrame
    ) -> List[Dict[str, Any]]:
        """Generate actionable optimization recommendations"""
        logger.info("Generating optimization recommendations...")
        
        recommendations = []
        
        # Bottleneck recommendations
        top_bottlenecks = bottlenecks.orderBy(F.desc("bottleneck_score")).limit(5).collect()
        for row in top_bottlenecks:
            recommendations.append({
                "type": "bottleneck",
                "task_type": row["task_type"],
                "agent": row["agent"],
                "recommendation": f"Optimize {row['task_type']} task - P90 duration is {row['bottleneck_score']:.1f}x above average",
                "impact": "high",
                "metrics": {
                    "p90_duration": row["p90_duration"],
                    "avg_retries": row["avg_retries"]
                }
            })
            
        # Agent assignment recommendations
        agent_improvements = agent_patterns.collect()
        for row in agent_improvements:
            recommendations.append({
                "type": "agent_assignment",
                "task_type": row["task_type"],
                "workflow_type": row["workflow_type"],
                "recommendation": f"Use {row['agent']} agent for {row['task_type']} tasks in {row['workflow_type']} workflows",
                "impact": "medium",
                "metrics": {
                    "success_rate": row["success_rate"],
                    "avg_duration": row["avg_duration"]
                }
            })
            
        # Success factor recommendations
        success_insights = success_factors.filter(
            F.abs(F.col("task_count_correlation")) > 0.3
        ).collect()
        
        for row in success_insights:
            if row["task_count_correlation"] < -0.3:
                recommendations.append({
                    "type": "workflow_design",
                    "workflow_type": row["workflow_type"],
                    "recommendation": f"Reduce task count in {row['workflow_type']} workflows - negative correlation with success",
                    "impact": "medium",
                    "metrics": {
                        "correlation": row["task_count_correlation"],
                        "success_rate": row["success_rate"]
                    }
                })
                
        return recommendations
        
    def publish_patterns_to_knowledge_graph(self, patterns: Dict[str, Any]):
        """Publish discovered patterns to knowledge graph"""
        try:
            operations = []
            
            # Create pattern nodes
            for pattern_type, pattern_data in patterns.items():
                pattern_id = f"pattern_{pattern_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                operations.append({
                    "operation": "upsert_vertex",
                    "label": "WorkflowPattern",
                    "id_key": "pattern_id",
                    "properties": {
                        "pattern_id": pattern_id,
                        "type": pattern_type,
                        "discovered_at": datetime.now().isoformat(),
                        "confidence": pattern_data.get("confidence", 0.0),
                        "support": pattern_data.get("support", 0.0),
                        "details": json.dumps(pattern_data)
                    }
                })
                
            # Send to knowledge graph
            with httpx.Client() as client:
                response = client.post(
                    f"{KNOWLEDGE_GRAPH_URL}/api/v1/ingest",
                    json={"operations": operations}
                )
                response.raise_for_status()
                
            logger.info(f"Published {len(operations)} patterns to knowledge graph")
            
        except Exception as e:
            logger.error(f"Failed to publish patterns to knowledge graph: {e}")
            
    def publish_results_to_pulsar(self, results: Dict[str, Any]):
        """Publish mining results to Pulsar"""
        try:
            message = {
                "event_type": "workflow_pattern_mining_completed",
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            
            self.producer.send(json.dumps(message).encode('utf-8'))
            logger.info("Published pattern mining results to Pulsar")
            
        except Exception as e:
            logger.error(f"Failed to publish results to Pulsar: {e}")
            
    def run(self):
        """Execute the pattern mining pipeline"""
        try:
            # Initialize connections
            self.initialize_pulsar()
            
            # Load workflow data
            df = self.load_workflow_data()
            
            if df.count() == 0:
                logger.warning("No workflow data found to analyze")
                return
                
            # Cache the dataframe for multiple operations
            df.cache()
            
            # Extract patterns
            sequence_patterns = self.extract_sequence_patterns(df)
            bottlenecks = self.detect_bottlenecks(df)
            success_factors = self.analyze_success_factors(df)
            agent_patterns = self.mine_agent_performance_patterns(df)
            
            # Generate recommendations
            recommendations = self.generate_optimization_recommendations(
                sequence_patterns,
                bottlenecks,
                success_factors,
                agent_patterns
            )
            
            # Prepare results
            results = {
                "sequence_patterns": sequence_patterns.limit(20).collect(),
                "bottlenecks": bottlenecks.collect(),
                "success_factors": success_factors.collect(),
                "agent_patterns": agent_patterns.collect(),
                "recommendations": recommendations,
                "metadata": {
                    "total_workflows_analyzed": df.count(),
                    "time_range": {
                        "start": df.agg(F.min("start_time")).collect()[0][0],
                        "end": df.agg(F.max("start_time")).collect()[0][0]
                    }
                }
            }
            
            # Publish results
            self.publish_patterns_to_knowledge_graph(results)
            self.publish_results_to_pulsar(results)
            
            # Log summary
            logger.info(f"Pattern mining completed. Found {len(recommendations)} recommendations")
            
        except Exception as e:
            logger.error(f"Pattern mining failed: {e}", exc_info=True)
            raise
        finally:
            if self.pulsar_client:
                self.pulsar_client.close()
                

def main():
    """Main entry point for the Spark job"""
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("WorkflowPatternMiner") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .getOrCreate()
        
    try:
        # Set log level
        spark.sparkContext.setLogLevel("WARN")
        
        # Run pattern mining
        miner = WorkflowPatternMiner(spark)
        miner.run()
        
    finally:
        spark.stop()
        

if __name__ == "__main__":
    main() 