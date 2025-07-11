# managerQ/app/core/observability_manager.py
import asyncio
import logging
from typing import List, Dict, Any, Set
from fastapi import WebSocket
import json
import random
import uuid

logger = logging.getLogger(__name__)

# --- NEW: Data structures for the world state ---
class Node:
    def __init__(self, id: str, label: str, node_type: str, parent: str = None):
        self.id = id
        self.label = label
        self.type = node_type
        self.parent = parent
        self.state = "idle"
        self.position = { # Random initial position
            "x": random.uniform(-50, 50),
            "y": random.uniform(-50, 50),
            "z": random.uniform(-10, 10)
        }
        self.metadata = {}

class Link:
    def __init__(self, source_id: str, target_id: str):
        self.id = f"{source_id}-{target_id}"
        self.source = source_id
        self.target = target_id
        self.state = "active"

class ObservabilityManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._world_state: Dict[str, Node] = {}
        self._links: Dict[str, Link] = {}
        self._is_running = False
        self._broadcaster_task: asyncio.Task = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New client connected. Total clients: {len(self.active_connections)}")

        # Send a snapshot of the current world state to the new client
        snapshot = {
            "type": "SNAPSHOT",
            "payload": {
                "nodes": [node.__dict__ for node in self._world_state.values()],
                "links": [link.__dict__ for link in self._links.values()]
            }
        }
        await websocket.send_text(json.dumps(snapshot))

        # Start the broadcaster if it's the first connection
        if not self._is_running and self.active_connections:
            self.start_broadcaster()

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.active_connections)}")
        # Stop the broadcaster if there are no more connections
        if not self.active_connections and self._is_running:
            self.stop_broadcaster()

    def start_broadcaster(self):
        """Starts the background task to broadcast updates."""
        if not self._is_running:
            self._is_running = True
            self._broadcaster_task = asyncio.create_task(self._broadcast_loop())
            logger.info("Observability broadcaster started.")

    def stop_broadcaster(self):
        """Stops the background broadcaster task."""
        if self._is_running and self._broadcaster_task:
            self._is_running = False
            self._broadcaster_task.cancel()
            logger.info("Observability broadcaster stopped.")

    async def _broadcast_loop(self):
        """The main loop for generating and broadcasting world state updates."""
        while self._is_running:
            try:
                # In a real system, this would be driven by actual events from the platform.
                # Here, we simulate events for demonstration.
                events = self._simulate_events()
                if events:
                    message = {"type": "TICK", "payload": events}
                    await self.broadcast(message)
                
                await asyncio.sleep(1) # Broadcast every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in broadcast loop", exc_info=True)
                await asyncio.sleep(5)

    def _simulate_events(self) -> List[Dict[str, Any]]:
        """Simulates events happening in the Q Platform for demonstration."""
        events = []

        # Chance to create a new agent
        if random.random() < 0.1 and len(self._world_state) < 20:
            agent_id = f"agent_{uuid.uuid4().hex[:6]}"
            new_agent = Node(id=agent_id, label=f"Agent-{random.choice(['devops', 'security'])}", node_type="agent")
            self._world_state[agent_id] = new_agent
            events.append({"event_type": "NODE_CREATED", "data": new_agent.__dict__})
            # Link it to a central manager node
            if "managerQ" not in self._world_state:
                 manager_node = Node(id="managerQ", label="ManagerQ", node_type="service")
                 self._world_state["managerQ"] = manager_node
                 events.append({"event_type": "NODE_CREATED", "data": manager_node.__dict__})
            new_link = Link(source_id=agent_id, target_id="managerQ")
            self._links[new_link.id] = new_link
            events.append({"event_type": "LINK_CREATED", "data": new_link.__dict__})
        
        # Chance for an existing agent to change state
        if self._world_state and random.random() < 0.5:
            agent_id = random.choice([n.id for n in self._world_state.values() if n.type == 'agent'])
            if agent_id:
                agent = self._world_state[agent_id]
                old_state = agent.state
                agent.state = random.choice(["thinking", "acting", "idle"])
                if agent.state != old_state:
                     events.append({"event_type": "NODE_STATE_CHANGED", "data": {"id": agent_id, "state": agent.state}})

        # Chance for agents to communicate
        if len([n for n in self._world_state.values() if n.type == 'agent']) > 1 and random.random() < 0.3:
            agents = [n.id for n in self._world_state.values() if n.type == 'agent']
            source, target = random.sample(agents, 2)
            # Create a temporary communication link
            comm_link = Link(source_id=source, target_id=target)
            events.append({"event_type": "LINK_PULSE", "data": {"id": comm_link.id, "source": source, "target": target}})

        # --- NEW: Simulate Swarm Intelligence Events ---
        if "logistics_swarm" in self._world_state and random.random() < 0.8:
            # 1. Simulate pheromone trail updates (for ACO)
            # In a real system, this would be a grid of values. We'll simulate a few points.
            pheromone_event = {
                "event_type": "PHEROMONE_UPDATE",
                "data": {
                    "trail_id": "route_A",
                    "points": [
                        {"x": random.uniform(-40, 40), "y": random.uniform(-40, 40), "intensity": random.random()},
                        {"x": random.uniform(-40, 40), "y": random.uniform(-40, 40), "intensity": random.random()},
                    ]
                }
            }
            events.append(pheromone_event)
            
            # 2. Simulate bee agent targeting a solution (for PSO)
            bee_agents = [n.id for n in self._world_state.values() if "bee" in n.label]
            if bee_agents:
                target_solution_id = random.choice(list(self._world_state.keys())) # Target a random node
                bee_agent_id = random.choice(bee_agents)
                target_event = {
                    "event_type": "AGENT_TARGET_CHANGED",
                    "data": {
                        "agent_id": bee_agent_id,
                        "target_id": target_solution_id
                    }
                }
                events.append(target_event)
        
        # Create the swarm for the first time
        elif "logistics_swarm" not in self._world_state:
             swarm_node = Node(id="logistics_swarm", label="Drone Swarm", node_type="swarm")
             self._world_state["logistics_swarm"] = swarm_node
             events.append({"event_type": "NODE_CREATED", "data": swarm_node.__dict__})
             # Create a few "bee" agents in the swarm
             for i in range(5):
                 bee_id = f"bee_agent_{i}"
                 bee_node = Node(id=bee_id, label=f"Bee-{i}", node_type="agent", parent="logistics_swarm")
                 self._world_state[bee_id] = bee_node
                 events.append({"event_type": "NODE_CREATED", "data": bee_node.__dict__})
        
        return events

    async def broadcast(self, data: dict):
        """Broadcasts data to all connected clients."""
        if not self.active_connections:
            return

        message = json.dumps(data)
        # Use asyncio.gather to send to all clients concurrently
        await asyncio.gather(
            *[connection.send_text(message) for connection in self.active_connections],
            return_exceptions=True
        )

# Singleton instance
observability_manager = ObservabilityManager() 