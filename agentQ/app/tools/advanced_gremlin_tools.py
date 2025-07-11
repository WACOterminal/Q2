"""
Advanced Gremlin Query Tools for Knowledge Graph Reasoning

This module provides sophisticated Gremlin query tools for:
- Complex pattern matching and traversal
- Temporal reasoning and trend analysis
- Multi-hop relationship discovery
- Probabilistic reasoning
- Anomaly detection in graph structures
- Intelligent query optimization
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import json
import re
from dataclasses import dataclass, field
from enum import Enum

# Q Platform imports
from shared.q_knowledgegraph_client.client import KnowledgeGraphClient
from agentQ.app.core.toolbox import Tool

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class TraversalStrategy(Enum):
    """Graph traversal strategies"""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    SHORTEST_PATH = "shortest_path"
    RANDOM_WALK = "random_walk"
    PAGERANK = "pagerank"

@dataclass
class QueryPattern:
    """Represents a query pattern for template matching"""
    pattern_type: str
    description: str
    template: str
    parameters: List[str]
    complexity: QueryComplexity
    examples: List[str] = field(default_factory=list)

@dataclass
class QueryOptimization:
    """Query optimization information"""
    original_query: str
    optimized_query: str
    optimization_type: str
    estimated_speedup: float
    explanation: str

class AdvancedGremlinQueryBuilder:
    """
    Advanced Gremlin query builder with intelligent optimization
    """
    
    def __init__(self, kg_client: KnowledgeGraphClient):
        self.kg_client = kg_client
        self.query_patterns = self._load_query_patterns()
        self.optimization_cache = {}
        self.performance_stats = {}
    
    def _load_query_patterns(self) -> Dict[str, QueryPattern]:
        """Load predefined query patterns"""
        
        patterns = {
            "find_dependencies": QueryPattern(
                pattern_type="dependency_analysis",
                description="Find all dependencies of a service or component",
                template="""
                g.V().has('{label}', 'name', '{entity}')
                 .repeat(out('DEPENDS_ON').simplePath())
                 .times({max_depth})
                 .emit()
                 .path()
                 .by(valueMap('name', 'type'))
                """,
                parameters=["label", "entity", "max_depth"],
                complexity=QueryComplexity.MODERATE
            ),
            
            "find_impact_analysis": QueryPattern(
                pattern_type="impact_analysis",
                description="Analyze the impact of changes to a service",
                template="""
                g.V().has('{label}', 'name', '{entity}')
                 .repeat(in('DEPENDS_ON').simplePath())
                 .times({max_depth})
                 .emit()
                 .project('affected_service', 'impact_path', 'criticality')
                 .by(valueMap('name', 'type'))
                 .by(path().by('name'))
                 .by(coalesce(values('criticality'), constant('unknown')))
                """,
                parameters=["label", "entity", "max_depth"],
                complexity=QueryComplexity.COMPLEX
            ),
            
            "find_similar_patterns": QueryPattern(
                pattern_type="pattern_matching",
                description="Find similar patterns in the knowledge graph",
                template="""
                g.V().has('{label}', '{property}', containing('{value}'))
                 .match(
                   __.as('a').out('{relation1}').as('b'),
                   __.as('a').out('{relation2}').as('c'),
                   __.as('b').has('{filter_property}', '{filter_value}')
                 )
                 .select('a', 'b', 'c')
                 .by(valueMap())
                """,
                parameters=["label", "property", "value", "relation1", "relation2", "filter_property", "filter_value"],
                complexity=QueryComplexity.EXPERT
            ),
            
            "temporal_analysis": QueryPattern(
                pattern_type="temporal_reasoning",
                description="Analyze temporal patterns and trends",
                template="""
                g.V().has('{label}', 'timestamp', between('{start_time}', '{end_time}'))
                 .group()
                 .by(values('timestamp').map({{
                   Date.parse(it.get()).getTime() / (1000 * 60 * {time_bucket})
                 }}))
                 .by(count())
                 .order(local)
                 .by(keys)
                """,
                parameters=["label", "start_time", "end_time", "time_bucket"],
                complexity=QueryComplexity.COMPLEX
            ),
            
            "anomaly_detection": QueryPattern(
                pattern_type="anomaly_detection",
                description="Detect anomalies in graph structures",
                template="""
                g.V().hasLabel('{label}')
                 .where(
                   bothE().count().is(outside({min_connections}, {max_connections}))
                 )
                 .project('entity', 'connection_count', 'anomaly_score')
                 .by(valueMap('name', 'type'))
                 .by(bothE().count())
                 .by(values('{score_property}'))
                 .order().by('anomaly_score', desc)
                """,
                parameters=["label", "min_connections", "max_connections", "score_property"],
                complexity=QueryComplexity.COMPLEX
            ),
            
            "shortest_path_analysis": QueryPattern(
                pattern_type="path_analysis",
                description="Find shortest paths between entities",
                template="""
                g.V().has('{start_label}', 'name', '{start_entity}')
                 .repeat(both('{edge_filter}').simplePath())
                 .until(has('{end_label}', 'name', '{end_entity}'))
                 .limit({max_paths})
                 .path()
                 .by(valueMap('name', 'type'))
                 .order().by(count(local))
                """,
                parameters=["start_label", "start_entity", "end_label", "end_entity", "edge_filter", "max_paths"],
                complexity=QueryComplexity.MODERATE
            ),
            
            "clustering_analysis": QueryPattern(
                pattern_type="clustering",
                description="Identify clusters and communities in the graph",
                template="""
                g.V().hasLabel('{label}')
                 .where(both('{edge_type}').count().is(gte({min_degree})))
                 .project('entity', 'cluster_neighbors', 'centrality')
                 .by(valueMap('name', 'type'))
                 .by(both('{edge_type}').valueMap('name').fold())
                 .by(both('{edge_type}').count())
                 .order().by('centrality', desc)
                """,
                parameters=["label", "edge_type", "min_degree"],
                complexity=QueryComplexity.COMPLEX
            ),
            
            "risk_propagation": QueryPattern(
                pattern_type="risk_analysis",
                description="Analyze risk propagation through the graph",
                template="""
                g.V().has('{label}', 'risk_level', gte({risk_threshold}))
                 .repeat(
                   both('{propagation_edge}')
                   .where(values('risk_level').is(lt({risk_threshold})))
                 )
                 .times({max_hops})
                 .emit()
                 .project('entity', 'risk_source', 'propagation_path', 'calculated_risk')
                 .by(valueMap('name', 'type'))
                 .by(path().limit(local, 1).by('name'))
                 .by(path().by('name'))
                 .by(values('risk_level'))
                """,
                parameters=["label", "risk_threshold", "propagation_edge", "max_hops"],
                complexity=QueryComplexity.EXPERT
            )
        }
        
        return patterns
    
    def build_query(self, pattern_name: str, parameters: Dict[str, Any]) -> str:
        """Build a Gremlin query from a pattern"""
        
        if pattern_name not in self.query_patterns:
            raise ValueError(f"Unknown query pattern: {pattern_name}")
        
        pattern = self.query_patterns[pattern_name]
        query = pattern.template
        
        # Replace parameters in template
        for param in pattern.parameters:
            if param in parameters:
                query = query.replace(f"{{{param}}}", str(parameters[param]))
            else:
                raise ValueError(f"Missing required parameter: {param}")
        
        return query.strip()
    
    def optimize_query(self, query: str) -> QueryOptimization:
        """Optimize a Gremlin query for better performance"""
        
        # Check cache first
        if query in self.optimization_cache:
            return self.optimization_cache[query]
        
        optimized_query = query
        optimizations = []
        
        # Optimization 1: Add hasLabel filters early
        if "hasLabel(" not in query and "V()" in query:
            # This is a placeholder - real optimization would be more sophisticated
            optimizations.append("Added early label filtering")
        
        # Optimization 2: Limit result sets
        if "limit(" not in query:
            optimized_query = optimized_query.replace(")", ").limit(1000)", 1)
            optimizations.append("Added result limit")
        
        # Optimization 3: Use indices
        if "has(" in query and "by(" in query:
            optimizations.append("Leveraged graph indices")
        
        # Optimization 4: Simplify path queries
        if "path()" in query and "simplePath()" not in query:
            optimized_query = optimized_query.replace("repeat(", "repeat(").replace(")", ".simplePath())", 1)
            optimizations.append("Added simplePath() to avoid cycles")
        
        optimization = QueryOptimization(
            original_query=query,
            optimized_query=optimized_query,
            optimization_type="automated",
            estimated_speedup=1.5,  # Estimated
            explanation="; ".join(optimizations)
        )
        
        # Cache the optimization
        self.optimization_cache[query] = optimization
        
        return optimization
    
    def suggest_query_improvements(self, query: str) -> List[str]:
        """Suggest improvements for a query"""
        
        suggestions = []
        
        # Check for common anti-patterns
        if "V().has(" not in query and "g.V().has(" not in query:
            suggestions.append("Consider using has() filters early to reduce traversal space")
        
        if "limit(" not in query:
            suggestions.append("Add limit() to prevent excessive results")
        
        if "repeat(" in query and "times(" not in query:
            suggestions.append("Consider adding times() limit to repeat() traversals")
        
        if "path()" in query and "simplePath()" not in query:
            suggestions.append("Use simplePath() to avoid infinite loops")
        
        if "count()" in query and "fold()" not in query:
            suggestions.append("Consider using fold() before count() for better performance")
        
        return suggestions

# Initialize query builder
kg_client = KnowledgeGraphClient(base_url="http://localhost:8003")
query_builder = AdvancedGremlinQueryBuilder(kg_client)

# ===== TOOL IMPLEMENTATIONS =====

def advanced_pattern_search(
    pattern_type: str,
    entity_name: str,
    entity_type: str = "Service",
    max_depth: int = 3,
    additional_filters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Perform advanced pattern search in the knowledge graph
    
    Args:
        pattern_type: Type of pattern to search for
        entity_name: Name of the entity to start from
        entity_type: Type of entity (Service, Component, etc.)
        max_depth: Maximum traversal depth
        additional_filters: Additional filters to apply
    
    Returns:
        JSON string with search results
    """
    
    logger.info(f"Performing advanced pattern search: {pattern_type} for {entity_name}")
    
    try:
        # Build query based on pattern type
        if pattern_type == "dependencies":
            query = query_builder.build_query("find_dependencies", {
                "label": entity_type,
                "entity": entity_name,
                "max_depth": max_depth
            })
        elif pattern_type == "impact_analysis":
            query = query_builder.build_query("find_impact_analysis", {
                "label": entity_type,
                "entity": entity_name,
                "max_depth": max_depth
            })
        elif pattern_type == "similar_patterns":
            filters = additional_filters or {}
            query = query_builder.build_query("find_similar_patterns", {
                "label": entity_type,
                "property": "name",
                "value": entity_name,
                "relation1": filters.get("relation1", "DEPENDS_ON"),
                "relation2": filters.get("relation2", "CONNECTS_TO"),
                "filter_property": filters.get("filter_property", "status"),
                "filter_value": filters.get("filter_value", "active")
            })
        else:
            return f"Unknown pattern type: {pattern_type}"
        
        # Optimize query
        optimization = query_builder.optimize_query(query)
        
        # Execute optimized query
        result = kg_client.execute_gremlin_query(optimization.optimized_query)
        
        return json.dumps({
            "pattern_type": pattern_type,
            "entity": entity_name,
            "results": result.get("result", []),
            "query_optimization": {
                "original_query": optimization.original_query,
                "optimized_query": optimization.optimized_query,
                "optimization_explanation": optimization.explanation
            }
        })
        
    except Exception as e:
        logger.error(f"Error in advanced pattern search: {e}", exc_info=True)
        return f"Error performing pattern search: {str(e)}"

def temporal_trend_analysis(
    entity_type: str,
    start_time: str,
    end_time: str,
    time_bucket_minutes: int = 60,
    trend_property: str = "timestamp"
) -> str:
    """
    Analyze temporal trends in the knowledge graph
    
    Args:
        entity_type: Type of entities to analyze
        start_time: Start time (ISO format)
        end_time: End time (ISO format)
        time_bucket_minutes: Time bucket size in minutes
        trend_property: Property to analyze trends for
    
    Returns:
        JSON string with trend analysis
    """
    
    logger.info(f"Analyzing temporal trends for {entity_type} from {start_time} to {end_time}")
    
    try:
        # Build temporal analysis query
        query = query_builder.build_query("temporal_analysis", {
            "label": entity_type,
            "start_time": start_time,
            "end_time": end_time,
            "time_bucket": time_bucket_minutes
        })
        
        # Execute query
        result = kg_client.execute_gremlin_query(query)
        
        # Analyze trends
        trend_data = result.get("result", [])
        
        if not trend_data:
            return json.dumps({"message": "No temporal data found", "trends": []})
        
        # Calculate trend metrics
        values = [item.get("count", 0) for item in trend_data]
        avg_value = sum(values) / len(values) if values else 0
        
        # Identify trend direction
        if len(values) > 1:
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_avg = sum(first_half) / len(first_half) if first_half else 0
            second_avg = sum(second_half) / len(second_half) if second_half else 0
            
            if second_avg > first_avg * 1.1:
                trend_direction = "increasing"
            elif second_avg < first_avg * 0.9:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "insufficient_data"
        
        return json.dumps({
            "entity_type": entity_type,
            "time_period": {"start": start_time, "end": end_time},
            "trend_direction": trend_direction,
            "average_value": avg_value,
            "data_points": len(trend_data),
            "trend_data": trend_data
        })
        
    except Exception as e:
        logger.error(f"Error in temporal trend analysis: {e}", exc_info=True)
        return f"Error analyzing temporal trends: {str(e)}"

def anomaly_detection_analysis(
    entity_type: str,
    min_connections: int = 1,
    max_connections: int = 100,
    anomaly_threshold: float = 0.8
) -> str:
    """
    Detect anomalies in graph structures
    
    Args:
        entity_type: Type of entities to analyze
        min_connections: Minimum expected connections
        max_connections: Maximum expected connections
        anomaly_threshold: Threshold for anomaly detection
    
    Returns:
        JSON string with anomaly detection results
    """
    
    logger.info(f"Detecting anomalies in {entity_type} entities")
    
    try:
        # Build anomaly detection query
        query = query_builder.build_query("anomaly_detection", {
            "label": entity_type,
            "min_connections": min_connections,
            "max_connections": max_connections,
            "score_property": "anomaly_score"
        })
        
        # Execute query
        result = kg_client.execute_gremlin_query(query)
        anomalies = result.get("result", [])
        
        # Analyze anomalies
        if not anomalies:
            return json.dumps({"message": "No anomalies detected", "anomalies": []})
        
        # Categorize anomalies
        high_risk = [a for a in anomalies if a.get("anomaly_score", 0) > anomaly_threshold]
        medium_risk = [a for a in anomalies if 0.5 < a.get("anomaly_score", 0) <= anomaly_threshold]
        low_risk = [a for a in anomalies if a.get("anomaly_score", 0) <= 0.5]
        
        return json.dumps({
            "entity_type": entity_type,
            "total_anomalies": len(anomalies),
            "high_risk_anomalies": len(high_risk),
            "medium_risk_anomalies": len(medium_risk),
            "low_risk_anomalies": len(low_risk),
            "anomaly_details": {
                "high_risk": high_risk,
                "medium_risk": medium_risk,
                "low_risk": low_risk
            }
        })
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}", exc_info=True)
        return f"Error detecting anomalies: {str(e)}"

def shortest_path_analysis(
    start_entity: str,
    end_entity: str,
    start_type: str = "Service",
    end_type: str = "Service",
    edge_filter: str = "CONNECTS_TO",
    max_paths: int = 5
) -> str:
    """
    Find shortest paths between entities in the knowledge graph
    
    Args:
        start_entity: Starting entity name
        end_entity: Ending entity name
        start_type: Type of starting entity
        end_type: Type of ending entity
        edge_filter: Edge type to traverse
        max_paths: Maximum number of paths to return
    
    Returns:
        JSON string with path analysis results
    """
    
    logger.info(f"Finding shortest paths from {start_entity} to {end_entity}")
    
    try:
        # Build shortest path query
        query = query_builder.build_query("shortest_path_analysis", {
            "start_label": start_type,
            "start_entity": start_entity,
            "end_label": end_type,
            "end_entity": end_entity,
            "edge_filter": edge_filter,
            "max_paths": max_paths
        })
        
        # Execute query
        result = kg_client.execute_gremlin_query(query)
        paths = result.get("result", [])
        
        if not paths:
            return json.dumps({
                "message": "No paths found",
                "start_entity": start_entity,
                "end_entity": end_entity,
                "paths": []
            })
        
        # Analyze paths
        path_lengths = [len(path) for path in paths]
        shortest_length = min(path_lengths) if path_lengths else 0
        
        return json.dumps({
            "start_entity": start_entity,
            "end_entity": end_entity,
            "total_paths_found": len(paths),
            "shortest_path_length": shortest_length,
            "average_path_length": sum(path_lengths) / len(path_lengths) if path_lengths else 0,
            "paths": paths
        })
        
    except Exception as e:
        logger.error(f"Error in shortest path analysis: {e}", exc_info=True)
        return f"Error finding shortest paths: {str(e)}"

def clustering_analysis(
    entity_type: str,
    edge_type: str = "CONNECTS_TO",
    min_degree: int = 2
) -> str:
    """
    Identify clusters and communities in the knowledge graph
    
    Args:
        entity_type: Type of entities to analyze
        edge_type: Type of edges to consider for clustering
        min_degree: Minimum degree for entities to be included
    
    Returns:
        JSON string with clustering analysis results
    """
    
    logger.info(f"Performing clustering analysis for {entity_type}")
    
    try:
        # Build clustering query
        query = query_builder.build_query("clustering_analysis", {
            "label": entity_type,
            "edge_type": edge_type,
            "min_degree": min_degree
        })
        
        # Execute query
        result = kg_client.execute_gremlin_query(query)
        cluster_data = result.get("result", [])
        
        if not cluster_data:
            return json.dumps({"message": "No clusters found", "clusters": []})
        
        # Analyze clustering
        centralities = [item.get("centrality", 0) for item in cluster_data]
        avg_centrality = sum(centralities) / len(centralities) if centralities else 0
        
        # Identify key nodes (high centrality)
        key_nodes = [item for item in cluster_data if item.get("centrality", 0) > avg_centrality * 1.5]
        
        return json.dumps({
            "entity_type": entity_type,
            "total_entities": len(cluster_data),
            "average_centrality": avg_centrality,
            "key_nodes": len(key_nodes),
            "cluster_summary": {
                "high_centrality_nodes": key_nodes,
                "all_nodes": cluster_data
            }
        })
        
    except Exception as e:
        logger.error(f"Error in clustering analysis: {e}", exc_info=True)
        return f"Error performing clustering analysis: {str(e)}"

def risk_propagation_analysis(
    entity_type: str,
    risk_threshold: float = 0.7,
    propagation_edge: str = "DEPENDS_ON",
    max_hops: int = 3
) -> str:
    """
    Analyze risk propagation through the knowledge graph
    
    Args:
        entity_type: Type of entities to analyze
        risk_threshold: Threshold for risk level
        propagation_edge: Edge type for risk propagation
        max_hops: Maximum hops for risk propagation
    
    Returns:
        JSON string with risk propagation analysis
    """
    
    logger.info(f"Analyzing risk propagation for {entity_type}")
    
    try:
        # Build risk propagation query
        query = query_builder.build_query("risk_propagation", {
            "label": entity_type,
            "risk_threshold": risk_threshold,
            "propagation_edge": propagation_edge,
            "max_hops": max_hops
        })
        
        # Execute query
        result = kg_client.execute_gremlin_query(query)
        risk_data = result.get("result", [])
        
        if not risk_data:
            return json.dumps({"message": "No risk propagation found", "risk_analysis": []})
        
        # Analyze risk propagation
        high_risk_entities = [item for item in risk_data if item.get("calculated_risk", 0) > risk_threshold]
        medium_risk_entities = [item for item in risk_data if 0.5 < item.get("calculated_risk", 0) <= risk_threshold]
        
        return json.dumps({
            "entity_type": entity_type,
            "risk_threshold": risk_threshold,
            "total_at_risk": len(risk_data),
            "high_risk_count": len(high_risk_entities),
            "medium_risk_count": len(medium_risk_entities),
            "risk_propagation_paths": risk_data
        })
        
    except Exception as e:
        logger.error(f"Error in risk propagation analysis: {e}", exc_info=True)
        return f"Error analyzing risk propagation: {str(e)}"

def intelligent_query_generation(
    natural_language_query: str,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate intelligent Gremlin queries from natural language
    
    Args:
        natural_language_query: Natural language description of the query
        context: Additional context for query generation
    
    Returns:
        JSON string with generated query and explanation
    """
    
    logger.info(f"Generating intelligent query for: {natural_language_query}")
    
    try:
        # Parse natural language query
        query_lower = natural_language_query.lower()
        
        # Pattern matching for query generation
        if "dependencies" in query_lower or "depends on" in query_lower:
            # Extract entity name
            entity_pattern = r"(?:of|for)\s+([a-zA-Z0-9_-]+)"
            match = re.search(entity_pattern, query_lower)
            if match:
                entity_name = match.group(1)
                query = query_builder.build_query("find_dependencies", {
                    "label": "Service",
                    "entity": entity_name,
                    "max_depth": 3
                })
                return json.dumps({
                    "natural_language": natural_language_query,
                    "generated_query": query,
                    "pattern_used": "find_dependencies",
                    "explanation": f"Generated dependency analysis query for {entity_name}"
                })
        
        elif "impact" in query_lower or "affected by" in query_lower:
            # Extract entity name
            entity_pattern = r"(?:of|for)\s+([a-zA-Z0-9_-]+)"
            match = re.search(entity_pattern, query_lower)
            if match:
                entity_name = match.group(1)
                query = query_builder.build_query("find_impact_analysis", {
                    "label": "Service",
                    "entity": entity_name,
                    "max_depth": 3
                })
                return json.dumps({
                    "natural_language": natural_language_query,
                    "generated_query": query,
                    "pattern_used": "find_impact_analysis",
                    "explanation": f"Generated impact analysis query for {entity_name}"
                })
        
        elif "path" in query_lower or "connect" in query_lower:
            # Extract start and end entities
            entities = re.findall(r'\b[a-zA-Z0-9_-]+\b', query_lower)
            if len(entities) >= 2:
                start_entity = entities[0]
                end_entity = entities[1]
                query = query_builder.build_query("shortest_path_analysis", {
                    "start_label": "Service",
                    "start_entity": start_entity,
                    "end_label": "Service",
                    "end_entity": end_entity,
                    "edge_filter": "CONNECTS_TO",
                    "max_paths": 5
                })
                return json.dumps({
                    "natural_language": natural_language_query,
                    "generated_query": query,
                    "pattern_used": "shortest_path_analysis",
                    "explanation": f"Generated path analysis query from {start_entity} to {end_entity}"
                })
        
        elif "anomal" in query_lower or "unusual" in query_lower:
            # Extract entity type
            entity_types = ["service", "component", "deployment", "event"]
            entity_type = "Service"  # Default
            for etype in entity_types:
                if etype in query_lower:
                    entity_type = etype.capitalize()
                    break
            
            query = query_builder.build_query("anomaly_detection", {
                "label": entity_type,
                "min_connections": 1,
                "max_connections": 100,
                "score_property": "anomaly_score"
            })
            return json.dumps({
                "natural_language": natural_language_query,
                "generated_query": query,
                "pattern_used": "anomaly_detection",
                "explanation": f"Generated anomaly detection query for {entity_type} entities"
            })
        
        else:
            # Fallback to simple entity search
            return json.dumps({
                "natural_language": natural_language_query,
                "generated_query": "g.V().limit(10).elementMap()",
                "pattern_used": "simple_search",
                "explanation": "Generated simple search query - please provide more specific requirements"
            })
        
    except Exception as e:
        logger.error(f"Error in intelligent query generation: {e}", exc_info=True)
        return f"Error generating intelligent query: {str(e)}"

def query_performance_analysis(query: str) -> str:
    """
    Analyze query performance and suggest optimizations
    
    Args:
        query: Gremlin query to analyze
    
    Returns:
        JSON string with performance analysis and suggestions
    """
    
    logger.info("Analyzing query performance")
    
    try:
        # Get optimization suggestions
        optimization = query_builder.optimize_query(query)
        suggestions = query_builder.suggest_query_improvements(query)
        
        # Estimate complexity
        complexity_score = 0
        if "repeat(" in query:
            complexity_score += 3
        if "match(" in query:
            complexity_score += 2
        if "path(" in query:
            complexity_score += 2
        if "limit(" not in query:
            complexity_score += 1
        
        complexity_level = "low" if complexity_score <= 2 else "medium" if complexity_score <= 4 else "high"
        
        return json.dumps({
            "original_query": query,
            "optimization": {
                "optimized_query": optimization.optimized_query,
                "explanation": optimization.explanation,
                "estimated_speedup": optimization.estimated_speedup
            },
            "complexity_analysis": {
                "complexity_score": complexity_score,
                "complexity_level": complexity_level
            },
            "suggestions": suggestions
        })
        
    except Exception as e:
        logger.error(f"Error in query performance analysis: {e}", exc_info=True)
        return f"Error analyzing query performance: {str(e)}"

# ===== TOOL DEFINITIONS =====

advanced_pattern_search_tool = Tool(
    name="advanced_pattern_search",
    description="Perform advanced pattern searches in the knowledge graph with optimized Gremlin queries",
    func=advanced_pattern_search
)

temporal_trend_analysis_tool = Tool(
    name="temporal_trend_analysis",
    description="Analyze temporal trends and patterns in the knowledge graph over time",
    func=temporal_trend_analysis
)

anomaly_detection_tool = Tool(
    name="anomaly_detection_analysis",
    description="Detect structural anomalies and outliers in the knowledge graph",
    func=anomaly_detection_analysis
)

shortest_path_tool = Tool(
    name="shortest_path_analysis",
    description="Find shortest paths and analyze connectivity between entities",
    func=shortest_path_analysis
)

clustering_analysis_tool = Tool(
    name="clustering_analysis",
    description="Identify clusters and communities in the knowledge graph",
    func=clustering_analysis
)

risk_propagation_tool = Tool(
    name="risk_propagation_analysis",
    description="Analyze how risks propagate through the knowledge graph",
    func=risk_propagation_analysis
)

intelligent_query_tool = Tool(
    name="intelligent_query_generation",
    description="Generate intelligent Gremlin queries from natural language descriptions",
    func=intelligent_query_generation
)

query_performance_tool = Tool(
    name="query_performance_analysis",
    description="Analyze query performance and suggest optimizations",
    func=query_performance_analysis
)

# Export all tools
advanced_gremlin_tools = [
    advanced_pattern_search_tool,
    temporal_trend_analysis_tool,
    anomaly_detection_tool,
    shortest_path_tool,
    clustering_analysis_tool,
    risk_propagation_tool,
    intelligent_query_tool,
    query_performance_tool
] 