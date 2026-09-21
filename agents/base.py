"""
Shared contract for the multi-agent legal reasoning pipeline.

Every agent (Fact Extraction -> Statute Retrieval -> Precedent Retrieval ->
Argument Analysis -> Prediction) reads from and writes to one shared `state`
dict. That is the same shape LangGraph (listed in requirements.txt) uses for
node functions, so wiring the agents into a LangGraph StateGraph later is a
mechanical step: `graph.add_node(agent.name, agent.run)`.
"""
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    name: str = "base_agent"

    @abstractmethod
    def run(self, state: dict) -> dict:
        """Read what this agent needs from `state`, return `state` with its output added."""
