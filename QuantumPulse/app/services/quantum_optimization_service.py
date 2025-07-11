import structlog
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
from qiskit.primitives import Sampler
from qiskit.circuit.library import RealAmplitudes
from docplex.mp.model import Model
import numpy as np

logger = structlog.get_logger(__name__)

class QuantumOptimizationService:
    """
    A service to solve optimization problems using quantum algorithms.
    """
    def __init__(self):
        # Setup a reusable VQE optimizer
        self._optimizer = MinimumEigenOptimizer(
            min_eigen_solver=SamplingVQE(
                sampler=Sampler(),
                ansatz=RealAmplitudes(),
            )
        )
        logger.info("QuantumOptimizationService initialized with VQE optimizer.")

    def solve_llm_routing(self, providers: list[dict]) -> dict:
        """
        Solves the LLM provider routing problem to balance cost and latency.

        Args:
            providers: A list of dicts, where each dict represents a provider
                       with 'name', 'cost_per_1k_tokens', and 'p90_latency_ms'.
        
        Returns:
            A dict containing the optimal provider choice and the reason.
        """
        logger.info("Solving LLM routing problem", providers=providers)

        # --- 1. Formulate the problem using Docplex ---
        mdl = Model("LLM Provider Selection")
        
        # Define binary variables: x[i] is 1 if we choose provider i, 0 otherwise.
        x = mdl.binary_var_list(len(providers), name="x")
        
        # --- 2. Define the objective function ---
        # We need to normalize cost and latency to combine them.
        costs = np.array([p['cost_per_1k_tokens'] for p in providers])
        latencies = np.array([p['p90_latency_ms'] for p in providers])
        
        # Normalize to a 0-1 scale. Add a small epsilon to avoid division by zero.
        norm_costs = (costs - costs.min()) / (costs.max() - costs.min() + 1e-6)
        norm_latencies = (latencies - latencies.min()) / (latencies.max() - latencies.min() + 1e-6)
        
        # Define weights for cost vs. latency. Let's say we care about them equally.
        w_cost = 0.5
        w_latency = 0.5
        
        objective = mdl.sum(x[i] * (w_cost * norm_costs[i] + w_latency * norm_latencies[i]) for i in range(len(providers)))
        mdl.minimize(objective)

        # --- 3. Define Constraints ---
        # We must choose exactly one provider.
        mdl.add_constraint(mdl.sum(x[i] for i in range(len(providers))) == 1)
        
        logger.info("Docplex model created", objective=str(objective), constraint="sum(x)==1")

        # --- 4. Translate to a QUBO ---
        qubo = from_docplex_mp(mdl)
        logger.info("Model translated to QUBO formulation.")

        # --- 5. Solve using the Quantum Optimizer ---
        try:
            result = self._optimizer.solve(qubo)
            
            if result.status.name == "SUCCESS":
                chosen_index = np.argmax(result.x)
                chosen_provider = providers[chosen_index]
                logger.info("Quantum optimization successful", chosen_provider=chosen_provider['name'])
                return {
                    "status": "SUCCESS",
                    "optimal_provider": chosen_provider,
                    "reason": f"Quantum solver found an optimal balance between cost and latency, selecting provider '{chosen_provider['name']}'.",
                    "full_result": str(result)
                }
            else:
                logger.error("Quantum optimization failed", status=result.status.name)
                return {"status": "FAILED", "reason": f"Quantum solver finished with status: {result.status.name}."}
        except Exception as e:
            logger.error("An exception occurred during quantum optimization", exc_info=True)
            return {"status": "ERROR", "reason": f"An unexpected error occurred: {str(e)}"}

# Singleton instance
quantum_optimization_service = QuantumOptimizationService() 