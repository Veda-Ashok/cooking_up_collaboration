from .base import Agent
from .human import HumanAgent
from .random import RandomAgent
from .auto import AutoAgent


AGENT_TYPES = (
    {"id": "human", "label": "Human", "class": HumanAgent},
    {"id": "random", "label": "Random", "class": RandomAgent},
    {"id": "auto", "label": "Auto", "class": AutoAgent},
)

AGENT_REGISTRY = {agent_type["id"]: agent_type for agent_type in AGENT_TYPES}


def get_agent_label(agent_id: str) -> str:
    return AGENT_REGISTRY[agent_id]["label"]


def get_agent_id(agent_info: str | dict[str, object]) -> str:
    if isinstance(agent_info, str):
        return agent_info

    agent_id = agent_info.get("id", agent_info.get("type"))
    if not isinstance(agent_id, str):
        raise ValueError(f"Invalid agent info: {agent_info}")
    return agent_id


def create_agent(agent_info: str | dict[str, object], player_num: int) -> Agent:
    agent_id = get_agent_id(agent_info)
    agent_class = AGENT_REGISTRY[agent_id]["class"]
    if agent_id == "human" and isinstance(agent_info, dict):
        return agent_class(player_num, input_type=agent_info.get("input_type"))
    return agent_class(player_num)
