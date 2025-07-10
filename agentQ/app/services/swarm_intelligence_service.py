"""
Swarm Intelligence Service

This service enables swarm intelligence for collective problem-solving:
- Distributed decision-making algorithms
- Emergent behavior coordination
- Collective knowledge aggregation
- Consensus building mechanisms
- Adaptive swarm topology
- Performance optimization through collective learning
"""

import logging
import asyncio
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import asdict, dataclass
from enum import Enum
import uuid
import numpy as np
from collections import defaultdict

# Q Platform imports
from shared.q_memory_schemas.memory_models import AgentMemory, MemoryType
from shared.q_analytics_schemas.models import PerformanceMetrics
from app.services.multi_agent_coordinator import (
    MultiAgentCoordinator, CoordinationTask, AgentCapability
)
from app.services.memory_service import MemoryService
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.pulsar_service import PulsarService
from app.services.ignite_service import IgniteService

logger = logging.getLogger(__name__)

class SwarmAlgorithm(Enum):
    """Swarm intelligence algorithms"""
    PARTICLE_SWARM = "particle_swarm"        # Particle Swarm Optimization
    ANT_COLONY = "ant_colony"                # Ant Colony Optimization
    BEE_COLONY = "bee_colony"                # Artificial Bee Colony
    FLOCKING = "flocking"                    # Boids flocking behavior
    CONSENSUS = "consensus"                  # Consensus algorithms
    DISTRIBUTED_VOTING = "distributed_voting" # Distributed voting

class SwarmBehavior(Enum):
    """Types of swarm behaviors"""
    EXPLORATION = "exploration"              # Exploring solution space
    EXPLOITATION = "exploitation"            # Exploiting known good solutions
    COORDINATION = "coordination"            # Coordinating agent actions
    CONSENSUS_BUILDING = "consensus_building" # Building consensus
    KNOWLEDGE_SHARING = "knowledge_sharing"  # Sharing knowledge
    ADAPTATION = "adaptation"                # Adapting to changes

class SwarmRole(Enum):
    """Roles in swarm intelligence"""
    SCOUT = "scout"                         # Explores new areas
    WORKER = "worker"                       # Processes tasks
    LEADER = "leader"                       # Guides swarm behavior
    FOLLOWER = "follower"                   # Follows others
    SPECIALIST = "specialist"               # Specialized function
    MESSENGER = "messenger"                 # Communicates between groups

@dataclass
class SwarmAgent:
    """Agent in swarm intelligence system"""
    agent_id: str
    position: List[float]  # Position in solution space
    velocity: List[float]  # Velocity vector
    fitness: float         # Current fitness score
    best_position: List[float]  # Personal best position
    best_fitness: float    # Personal best fitness
    role: SwarmRole
    neighborhood: Set[str] # Connected agents
    influence: float       # Influence in swarm
    energy: float         # Energy level
    last_update: datetime

@dataclass
class SwarmProblem:
    """Problem for swarm to solve"""
    problem_id: str
    problem_type: str
    description: str
    objective_function: str  # How to evaluate solutions
    constraints: List[Dict[str, Any]]
    solution_space_dimensions: int
    solution_bounds: List[Tuple[float, float]]  # Min/max for each dimension
    target_fitness: Optional[float]
    max_iterations: int
    convergence_criteria: Dict[str, Any]
    created_at: datetime
    
@dataclass
class SwarmSolution:
    """Solution found by swarm"""
    solution_id: str
    problem_id: str
    solution_vector: List[float]
    fitness_score: float
    confidence: float
    contributing_agents: List[str]
    generation: int
    discovered_at: datetime
    validation_status: str

@dataclass
class SwarmMessage:
    """Message exchanged between agents in swarm"""
    message_id: str
    sender_id: str
    receiver_ids: List[str]  # Can broadcast to multiple
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    ttl: int  # Time to live

class SwarmIntelligenceService:
    """
    Service for implementing swarm intelligence algorithms
    """
    
    def __init__(self):
        self.coordinator = MultiAgentCoordinator()
        self.memory_service = MemoryService()
        self.knowledge_graph = KnowledgeGraphService()
        self.pulsar_service = PulsarService()
        self.ignite_service = IgniteService()
        
        # Swarm state
        self.active_swarms: Dict[str, Dict[str, SwarmAgent]] = {}
        self.swarm_problems: Dict[str, SwarmProblem] = {}
        self.swarm_solutions: Dict[str, List[SwarmSolution]] = {}
        self.global_best_solutions: Dict[str, SwarmSolution] = {}
        
        # Communication infrastructure
        self.message_queues: Dict[str, List[SwarmMessage]] = defaultdict(list)
        self.pheromone_trails: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Algorithm parameters
        self.swarm_params = {
            "particle_swarm": {
                "inertia_weight": 0.9,
                "cognitive_factor": 2.0,
                "social_factor": 2.0,
                "max_velocity": 1.0
            },
            "ant_colony": {
                "pheromone_evaporation": 0.1,
                "pheromone_deposit": 1.0,
                "alpha": 1.0,  # Pheromone importance
                "beta": 2.0    # Heuristic importance
            },
            "bee_colony": {
                "scout_percentage": 0.1,
                "elite_sites": 3,
                "best_sites": 10,
                "recruited_bees": 20
            }
        }
        
        # Performance tracking
        self.swarm_metrics = {
            "problems_solved": 0,
            "average_convergence_time": 0.0,
            "solution_quality": 0.0,
            "swarm_efficiency": 0.0
        }
    
    async def initialize(self):
        """Initialize the swarm intelligence service"""
        logger.info("Initializing Swarm Intelligence Service")
        
        # Setup Pulsar topics for swarm communication
        await self._setup_swarm_topics()
        
        # Start swarm coordination loops
        asyncio.create_task(self._swarm_coordination_loop())
        asyncio.create_task(self._message_processing_loop())
        asyncio.create_task(self._pheromone_update_loop())
        
        logger.info("Swarm Intelligence Service initialized successfully")
    
    # ===== SWARM PROBLEM SOLVING =====
    
    async def solve_problem_with_swarm(
        self,
        problem_description: str,
        problem_type: str,
        swarm_size: int = 20,
        algorithm: SwarmAlgorithm = SwarmAlgorithm.PARTICLE_SWARM,
        max_iterations: int = 1000,
        solution_dimensions: int = 10
    ) -> SwarmSolution:
        """
        Solve a problem using swarm intelligence
        
        Args:
            problem_description: Description of the problem
            problem_type: Type of problem
            swarm_size: Number of agents in swarm
            algorithm: Swarm algorithm to use
            max_iterations: Maximum iterations
            solution_dimensions: Dimensions of solution space
            
        Returns:
            Best solution found by swarm
        """
        logger.info(f"Starting swarm problem solving with {swarm_size} agents")
        
        # Create problem definition
        problem = SwarmProblem(
            problem_id=f"problem_{uuid.uuid4().hex[:12]}",
            problem_type=problem_type,
            description=problem_description,
            objective_function=self._get_objective_function(problem_type),
            constraints=[],
            solution_space_dimensions=solution_dimensions,
            solution_bounds=[(-10.0, 10.0)] * solution_dimensions,  # Default bounds
            target_fitness=None,
            max_iterations=max_iterations,
            convergence_criteria={"stagnation_threshold": 100},
            created_at=datetime.utcnow()
        )
        
        self.swarm_problems[problem.problem_id] = problem
        
        # Initialize swarm
        swarm = await self._initialize_swarm(problem, swarm_size, algorithm)
        self.active_swarms[problem.problem_id] = swarm
        
        # Run swarm algorithm
        best_solution = await self._run_swarm_algorithm(problem, swarm, algorithm)
        
        # Store solution
        if problem.problem_id not in self.swarm_solutions:
            self.swarm_solutions[problem.problem_id] = []
        self.swarm_solutions[problem.problem_id].append(best_solution)
        self.global_best_solutions[problem.problem_id] = best_solution
        
        # Publish results
        await self.pulsar_service.publish(
            "q.swarm.problem.solved",
            {
                "problem_id": problem.problem_id,
                "algorithm": algorithm.value,
                "swarm_size": swarm_size,
                "best_fitness": best_solution.fitness_score,
                "generations": best_solution.generation,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Swarm problem solved with fitness: {best_solution.fitness_score}")
        return best_solution
    
    async def _initialize_swarm(
        self,
        problem: SwarmProblem,
        swarm_size: int,
        algorithm: SwarmAlgorithm
    ) -> Dict[str, SwarmAgent]:
        """Initialize swarm agents for problem solving"""
        swarm = {}
        
        for i in range(swarm_size):
            agent_id = f"swarm_agent_{problem.problem_id}_{i}"
            
            # Random initial position within bounds
            position = []
            for bound in problem.solution_bounds:
                pos = np.random.uniform(bound[0], bound[1])
                position.append(pos)
            
            # Random initial velocity
            velocity = [np.random.uniform(-1, 1) for _ in range(problem.solution_space_dimensions)]
            
            # Evaluate initial fitness
            fitness = await self._evaluate_fitness(position, problem)
            
            # Assign role based on algorithm
            role = self._assign_swarm_role(i, swarm_size, algorithm)
            
            agent = SwarmAgent(
                agent_id=agent_id,
                position=position,
                velocity=velocity,
                fitness=fitness,
                best_position=position.copy(),
                best_fitness=fitness,
                role=role,
                neighborhood=set(),
                influence=1.0,
                energy=1.0,
                last_update=datetime.utcnow()
            )
            
            swarm[agent_id] = agent
        
        # Establish neighborhood connections
        await self._establish_swarm_topology(swarm, algorithm)
        
        return swarm
    
    async def _run_swarm_algorithm(
        self,
        problem: SwarmProblem,
        swarm: Dict[str, SwarmAgent],
        algorithm: SwarmAlgorithm
    ) -> SwarmSolution:
        """Run the specified swarm algorithm"""
        logger.info(f"Running {algorithm.value} algorithm")
        
        if algorithm == SwarmAlgorithm.PARTICLE_SWARM:
            return await self._run_particle_swarm_optimization(problem, swarm)
        elif algorithm == SwarmAlgorithm.ANT_COLONY:
            return await self._run_ant_colony_optimization(problem, swarm)
        elif algorithm == SwarmAlgorithm.BEE_COLONY:
            return await self._run_bee_colony_optimization(problem, swarm)
        elif algorithm == SwarmAlgorithm.CONSENSUS:
            return await self._run_consensus_algorithm(problem, swarm)
        else:
            return await self._run_generic_swarm_algorithm(problem, swarm)
    
    # ===== PARTICLE SWARM OPTIMIZATION =====
    
    async def _run_particle_swarm_optimization(
        self,
        problem: SwarmProblem,
        swarm: Dict[str, SwarmAgent]
    ) -> SwarmSolution:
        """Run Particle Swarm Optimization algorithm"""
        logger.debug("Running Particle Swarm Optimization")
        
        params = self.swarm_params["particle_swarm"]
        global_best_agent = max(swarm.values(), key=lambda x: x.fitness)
        global_best_position = global_best_agent.best_position.copy()
        global_best_fitness = global_best_agent.best_fitness
        
        generation = 0
        stagnation_count = 0
        
        for iteration in range(problem.max_iterations):
            for agent in swarm.values():
                # Update velocity
                for d in range(problem.solution_space_dimensions):
                    r1, r2 = np.random.random(), np.random.random()
                    
                    cognitive_component = (
                        params["cognitive_factor"] * r1 * 
                        (agent.best_position[d] - agent.position[d])
                    )
                    
                    social_component = (
                        params["social_factor"] * r2 * 
                        (global_best_position[d] - agent.position[d])
                    )
                    
                    agent.velocity[d] = (
                        params["inertia_weight"] * agent.velocity[d] +
                        cognitive_component + social_component
                    )
                    
                    # Limit velocity
                    if abs(agent.velocity[d]) > params["max_velocity"]:
                        agent.velocity[d] = np.sign(agent.velocity[d]) * params["max_velocity"]
                
                # Update position
                for d in range(problem.solution_space_dimensions):
                    agent.position[d] += agent.velocity[d]
                    
                    # Keep within bounds
                    bound = problem.solution_bounds[d]
                    agent.position[d] = max(bound[0], min(bound[1], agent.position[d]))
                
                # Evaluate fitness
                agent.fitness = await self._evaluate_fitness(agent.position, problem)
                
                # Update personal best
                if agent.fitness > agent.best_fitness:
                    agent.best_position = agent.position.copy()
                    agent.best_fitness = agent.fitness
                
                # Update global best
                if agent.fitness > global_best_fitness:
                    global_best_position = agent.position.copy()
                    global_best_fitness = agent.fitness
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            
            generation = iteration
            
            # Check convergence
            if stagnation_count > problem.convergence_criteria.get("stagnation_threshold", 100):
                logger.info(f"PSO converged after {iteration} iterations")
                break
            
            # Adaptive parameters
            if iteration % 100 == 0:
                params["inertia_weight"] *= 0.95  # Decrease inertia over time
        
        return SwarmSolution(
            solution_id=f"solution_{uuid.uuid4().hex[:12]}",
            problem_id=problem.problem_id,
            solution_vector=global_best_position,
            fitness_score=global_best_fitness,
            confidence=self._calculate_solution_confidence(swarm, global_best_position),
            contributing_agents=list(swarm.keys()),
            generation=generation,
            discovered_at=datetime.utcnow(),
            validation_status="pending"
        )
    
    # ===== ANT COLONY OPTIMIZATION =====
    
    async def _run_ant_colony_optimization(
        self,
        problem: SwarmProblem,
        swarm: Dict[str, SwarmAgent]
    ) -> SwarmSolution:
        """Run Ant Colony Optimization algorithm"""
        logger.debug("Running Ant Colony Optimization")
        
        params = self.swarm_params["ant_colony"]
        best_solution = None
        best_fitness = float('-inf')
        
        # Initialize pheromone trails
        pheromone_key = f"pheromones_{problem.problem_id}"
        self.pheromone_trails[pheromone_key] = {}
        
        for iteration in range(problem.max_iterations):
            # Each ant constructs a solution
            for agent in swarm.values():
                solution_path = await self._construct_ant_solution(agent, problem, pheromone_key)
                fitness = await self._evaluate_fitness(solution_path, problem)
                
                agent.position = solution_path
                agent.fitness = fitness
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution_path.copy()
            
            # Update pheromone trails
            await self._update_pheromone_trails(swarm, pheromone_key, params)
            
            # Evaporate pheromones
            for key in self.pheromone_trails[pheromone_key]:
                self.pheromone_trails[pheromone_key][key] *= (1 - params["pheromone_evaporation"])
        
        return SwarmSolution(
            solution_id=f"solution_{uuid.uuid4().hex[:12]}",
            problem_id=problem.problem_id,
            solution_vector=best_solution,
            fitness_score=best_fitness,
            confidence=self._calculate_solution_confidence(swarm, best_solution),
            contributing_agents=list(swarm.keys()),
            generation=problem.max_iterations,
            discovered_at=datetime.utcnow(),
            validation_status="pending"
        )
    
    async def _construct_ant_solution(
        self,
        ant: SwarmAgent,
        problem: SwarmProblem,
        pheromone_key: str
    ) -> List[float]:
        """Construct solution for ant using pheromone trails and heuristics"""
        solution = []
        
        for dimension in range(problem.solution_space_dimensions):
            # Discretize the dimension space for pheromone lookup
            bound = problem.solution_bounds[dimension]
            num_steps = 100  # Discretization steps
            step_size = (bound[1] - bound[0]) / num_steps
            
            # Calculate probabilities based on pheromone and heuristic information
            probabilities = []
            values = []
            
            for step in range(num_steps):
                value = bound[0] + step * step_size
                values.append(value)
                
                # Get pheromone level
                pheromone_level = self.pheromone_trails[pheromone_key].get(
                    f"{dimension}_{step}", 0.1
                )
                
                # Heuristic information (distance to current best)
                heuristic = 1.0 / (1.0 + abs(value))  # Simple heuristic
                
                # Combine pheromone and heuristic
                params = self.swarm_params["ant_colony"]
                probability = (pheromone_level ** params["alpha"]) * (heuristic ** params["beta"])
                probabilities.append(probability)
            
            # Normalize probabilities
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
            else:
                probabilities = [1.0 / len(probabilities)] * len(probabilities)
            
            # Select value based on probability
            selected_index = np.random.choice(len(values), p=probabilities)
            solution.append(values[selected_index])
        
        return solution
    
    async def _update_pheromone_trails(
        self,
        swarm: Dict[str, SwarmAgent],
        pheromone_key: str,
        params: Dict[str, float]
    ):
        """Update pheromone trails based on ant solutions"""
        for agent in swarm.values():
            # Deposit pheromone proportional to solution quality
            pheromone_amount = params["pheromone_deposit"] * agent.fitness
            
            for dimension, value in enumerate(agent.position):
                # Convert continuous value to discrete step
                # (This is a simplified approach)
                step = int(value * 10) % 100  # Simple discretization
                trail_key = f"{dimension}_{step}"
                
                current_pheromone = self.pheromone_trails[pheromone_key].get(trail_key, 0.1)
                self.pheromone_trails[pheromone_key][trail_key] = current_pheromone + pheromone_amount
    
    # ===== CONSENSUS ALGORITHM =====
    
    async def _run_consensus_algorithm(
        self,
        problem: SwarmProblem,
        swarm: Dict[str, SwarmAgent]
    ) -> SwarmSolution:
        """Run consensus-based swarm algorithm"""
        logger.debug("Running Consensus Algorithm")
        
        consensus_rounds = problem.max_iterations // 10  # Reduce rounds for consensus
        
        for round_num in range(consensus_rounds):
            # Each agent proposes a solution
            proposals = {}
            for agent in swarm.values():
                # Agent proposes a solution based on local knowledge
                proposal = await self._generate_agent_proposal(agent, problem)
                proposals[agent.agent_id] = proposal
                agent.position = proposal
                agent.fitness = await self._evaluate_fitness(proposal, problem)
            
            # Agents communicate and influence each other
            await self._consensus_communication_round(swarm, proposals)
            
            # Update agent positions based on consensus
            await self._update_consensus_positions(swarm, proposals)
        
        # Find best consensus solution
        best_agent = max(swarm.values(), key=lambda x: x.fitness)
        
        return SwarmSolution(
            solution_id=f"solution_{uuid.uuid4().hex[:12]}",
            problem_id=problem.problem_id,
            solution_vector=best_agent.position,
            fitness_score=best_agent.fitness,
            confidence=self._calculate_solution_confidence(swarm, best_agent.position),
            contributing_agents=list(swarm.keys()),
            generation=consensus_rounds,
            discovered_at=datetime.utcnow(),
            validation_status="pending"
        )
    
    # ===== SWARM COMMUNICATION =====
    
    async def send_swarm_message(
        self,
        sender_id: str,
        receiver_ids: List[str],
        message_type: str,
        content: Dict[str, Any],
        ttl: int = 300
    ):
        """Send message between swarm agents"""
        message = SwarmMessage(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            sender_id=sender_id,
            receiver_ids=receiver_ids,
            message_type=message_type,
            content=content,
            timestamp=datetime.utcnow(),
            ttl=ttl
        )
        
        # Add to message queues
        for receiver_id in receiver_ids:
            self.message_queues[receiver_id].append(message)
        
        # Publish to Pulsar for distribution
        await self.pulsar_service.publish(
            f"q.swarm.message.{message_type}",
            asdict(message)
        )
    
    async def _consensus_communication_round(
        self,
        swarm: Dict[str, SwarmAgent],
        proposals: Dict[str, List[float]]
    ):
        """One round of consensus communication"""
        for agent in swarm.values():
            # Agent shares its proposal with neighbors
            neighbor_proposals = []
            for neighbor_id in agent.neighborhood:
                if neighbor_id in proposals:
                    neighbor_proposals.append(proposals[neighbor_id])
            
            # Send proposal to neighbors
            if agent.neighborhood:
                await self.send_swarm_message(
                    agent.agent_id,
                    list(agent.neighborhood),
                    "proposal",
                    {
                        "proposal": proposals[agent.agent_id],
                        "fitness": agent.fitness,
                        "confidence": agent.influence
                    }
                )
    
    # ===== HELPER METHODS =====
    
    def _get_objective_function(self, problem_type: str) -> str:
        """Get objective function for problem type"""
        functions = {
            "optimization": "maximize_fitness",
            "classification": "maximize_accuracy",
            "clustering": "minimize_distance",
            "scheduling": "minimize_makespan",
            "routing": "minimize_path_length"
        }
        return functions.get(problem_type, "maximize_fitness")
    
    async def _evaluate_fitness(self, solution: List[float], problem: SwarmProblem) -> float:
        """Evaluate fitness of a solution"""
        # This is a placeholder - would implement actual fitness evaluation
        # For now, using a simple sphere function as test
        return -sum(x**2 for x in solution)  # Negative because we want to minimize
    
    def _assign_swarm_role(self, agent_index: int, swarm_size: int, algorithm: SwarmAlgorithm) -> SwarmRole:
        """Assign role to swarm agent based on algorithm"""
        if algorithm == SwarmAlgorithm.BEE_COLONY:
            if agent_index < swarm_size * 0.1:
                return SwarmRole.SCOUT
            elif agent_index < swarm_size * 0.3:
                return SwarmRole.LEADER
            else:
                return SwarmRole.WORKER
        elif algorithm == SwarmAlgorithm.ANT_COLONY:
            return SwarmRole.WORKER
        else:
            if agent_index == 0:
                return SwarmRole.LEADER
            else:
                return SwarmRole.FOLLOWER
    
    async def _establish_swarm_topology(
        self,
        swarm: Dict[str, SwarmAgent],
        algorithm: SwarmAlgorithm
    ):
        """Establish neighborhood topology for swarm"""
        agents = list(swarm.values())
        
        if algorithm == SwarmAlgorithm.PARTICLE_SWARM:
            # Ring topology for PSO
            for i, agent in enumerate(agents):
                left_neighbor = agents[(i - 1) % len(agents)]
                right_neighbor = agents[(i + 1) % len(agents)]
                agent.neighborhood.add(left_neighbor.agent_id)
                agent.neighborhood.add(right_neighbor.agent_id)
        
        elif algorithm == SwarmAlgorithm.CONSENSUS:
            # Fully connected for consensus
            for agent in agents:
                agent.neighborhood = {other.agent_id for other in agents if other.agent_id != agent.agent_id}
        
        else:
            # Random topology
            for agent in agents:
                num_neighbors = min(5, len(agents) - 1)
                possible_neighbors = [a for a in agents if a.agent_id != agent.agent_id]
                neighbors = np.random.choice(possible_neighbors, num_neighbors, replace=False)
                agent.neighborhood = {n.agent_id for n in neighbors}
    
    def _calculate_solution_confidence(
        self,
        swarm: Dict[str, SwarmAgent],
        solution: List[float]
    ) -> float:
        """Calculate confidence in solution based on swarm agreement"""
        if not swarm:
            return 0.0
        
        # Calculate average distance to solution
        distances = []
        for agent in swarm.values():
            distance = math.sqrt(sum((a - b)**2 for a, b in zip(agent.position, solution)))
            distances.append(distance)
        
        avg_distance = sum(distances) / len(distances)
        
        # Convert distance to confidence (lower distance = higher confidence)
        confidence = 1.0 / (1.0 + avg_distance)
        return confidence
    
    async def _generate_agent_proposal(
        self,
        agent: SwarmAgent,
        problem: SwarmProblem
    ) -> List[float]:
        """Generate proposal from agent based on its knowledge"""
        # Use agent's current best position as starting point
        proposal = agent.best_position.copy()
        
        # Add some exploration
        for i in range(len(proposal)):
            noise = np.random.normal(0, 0.1)  # Small random perturbation
            proposal[i] += noise
            
            # Keep within bounds
            bound = problem.solution_bounds[i]
            proposal[i] = max(bound[0], min(bound[1], proposal[i]))
        
        return proposal
    
    async def _update_consensus_positions(
        self,
        swarm: Dict[str, SwarmAgent],
        proposals: Dict[str, List[float]]
    ):
        """Update agent positions based on consensus"""
        for agent in swarm.values():
            # Calculate weighted average of neighbor proposals
            neighbor_proposals = []
            weights = []
            
            for neighbor_id in agent.neighborhood:
                if neighbor_id in swarm:
                    neighbor = swarm[neighbor_id]
                    neighbor_proposals.append(proposals[neighbor_id])
                    weights.append(neighbor.influence)
            
            if neighbor_proposals:
                # Weighted average
                total_weight = sum(weights)
                if total_weight > 0:
                    new_position = []
                    for dim in range(len(agent.position)):
                        weighted_sum = sum(
                            prop[dim] * weight 
                            for prop, weight in zip(neighbor_proposals, weights)
                        )
                        new_position.append(weighted_sum / total_weight)
                    
                    # Update position (blend with current position)
                    blend_factor = 0.3
                    for i in range(len(agent.position)):
                        agent.position[i] = (
                            (1 - blend_factor) * agent.position[i] +
                            blend_factor * new_position[i]
                        )
    
    # ===== BACKGROUND TASKS =====
    
    async def _swarm_coordination_loop(self):
        """Main coordination loop for swarms"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Monitor swarm health
                for problem_id, swarm in self.active_swarms.items():
                    await self._monitor_swarm_health(problem_id, swarm)
                
                # Update swarm metrics
                await self._update_swarm_metrics()
                
            except Exception as e:
                logger.error(f"Error in swarm coordination loop: {e}")
    
    async def _message_processing_loop(self):
        """Process swarm messages"""
        while True:
            try:
                await asyncio.sleep(1)  # Process messages frequently
                
                current_time = datetime.utcnow()
                
                # Process messages for each agent
                for agent_id, messages in self.message_queues.items():
                    # Remove expired messages
                    valid_messages = [
                        msg for msg in messages
                        if (current_time - msg.timestamp).total_seconds() < msg.ttl
                    ]
                    self.message_queues[agent_id] = valid_messages
                    
                    # Process valid messages
                    for message in valid_messages:
                        await self._process_swarm_message(agent_id, message)
                
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
    
    async def _pheromone_update_loop(self):
        """Update pheromone trails periodically"""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Evaporate pheromones
                for trail_key, trail in self.pheromone_trails.items():
                    for location, level in trail.items():
                        trail[location] = level * 0.99  # Slow evaporation
                
            except Exception as e:
                logger.error(f"Error in pheromone update loop: {e}")
    
    async def _monitor_swarm_health(self, problem_id: str, swarm: Dict[str, SwarmAgent]):
        """Monitor health of swarm agents"""
        current_time = datetime.utcnow()
        
        for agent in swarm.values():
            # Check if agent is stale
            time_since_update = (current_time - agent.last_update).total_seconds()
            if time_since_update > 300:  # 5 minutes
                # Agent is stale, reduce its energy
                agent.energy *= 0.9
                
                if agent.energy < 0.1:
                    # Remove or replace stale agent
                    logger.warning(f"Agent {agent.agent_id} is inactive, reducing influence")
                    agent.influence *= 0.5
    
    async def _process_swarm_message(self, agent_id: str, message: SwarmMessage):
        """Process a swarm message for an agent"""
        # Find which swarm this agent belongs to
        for problem_id, swarm in self.active_swarms.items():
            if agent_id in swarm:
                agent = swarm[agent_id]
                
                if message.message_type == "proposal":
                    # Process proposal message
                    proposal = message.content.get("proposal", [])
                    fitness = message.content.get("fitness", 0.0)
                    
                    # Update agent's knowledge
                    if fitness > agent.fitness:
                        # Learn from better solution
                        blend_factor = 0.1
                        for i in range(len(agent.position)):
                            if i < len(proposal):
                                agent.position[i] = (
                                    (1 - blend_factor) * agent.position[i] +
                                    blend_factor * proposal[i]
                                )
                
                break
    
    async def _update_swarm_metrics(self):
        """Update swarm performance metrics"""
        # Calculate metrics across all active swarms
        total_agents = sum(len(swarm) for swarm in self.active_swarms.values())
        active_problems = len(self.active_swarms)
        
        # Update global metrics
        self.swarm_metrics["active_problems"] = active_problems
        self.swarm_metrics["total_agents"] = total_agents
    
    # ===== PLACEHOLDER METHODS =====
    
    async def _run_bee_colony_optimization(
        self,
        problem: SwarmProblem,
        swarm: Dict[str, SwarmAgent]
    ) -> SwarmSolution:
        """Run Artificial Bee Colony algorithm"""
        # Placeholder - would implement full ABC algorithm
        return await self._run_generic_swarm_algorithm(problem, swarm)
    
    async def _run_generic_swarm_algorithm(
        self,
        problem: SwarmProblem,
        swarm: Dict[str, SwarmAgent]
    ) -> SwarmSolution:
        """Run generic swarm algorithm"""
        best_agent = max(swarm.values(), key=lambda x: x.fitness)
        
        return SwarmSolution(
            solution_id=f"solution_{uuid.uuid4().hex[:12]}",
            problem_id=problem.problem_id,
            solution_vector=best_agent.position,
            fitness_score=best_agent.fitness,
            confidence=0.5,
            contributing_agents=list(swarm.keys()),
            generation=0,
            discovered_at=datetime.utcnow(),
            validation_status="pending"
        )
    
    async def _setup_swarm_topics(self):
        """Setup Pulsar topics for swarm communication"""
        topics = [
            "q.swarm.problem.solved",
            "q.swarm.message.proposal",
            "q.swarm.message.consensus",
            "q.swarm.coordination"
        ]
        
        for topic in topics:
            await self.pulsar_service.ensure_topic(topic)

# Global service instance
swarm_intelligence_service = SwarmIntelligenceService() 