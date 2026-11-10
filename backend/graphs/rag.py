from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any

from graphs.rag_agent import agent

from langgraph.types import interrupt
from langgraph.graph import MessagesState
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import SystemMessage

load_dotenv()


def _load_graph_prompt(graph_dir: Path, filename: str) -> str:
    path = graph_dir / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


_GRAPH_DIR = Path(__file__).parent
SYSTEM_PROMPT: str = _load_graph_prompt(_GRAPH_DIR, "SOUL.md")


class ChatState(MessagesState):
    pass


def init_node(state: ChatState) -> dict:
    return {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
        ]
    }


async def user_input_node(state: ChatState) -> Dict[str, Any]:
    interrupt({"reason": f"Waiting for user input"})
    return


graph = StateGraph(ChatState)

graph.add_node("init", init_node)
graph.add_node("agent", agent)
graph.add_node("user_input", user_input_node)

graph.add_edge(START, "init")
graph.add_edge("init", "user_input")
graph.add_edge("user_input", "agent")
graph.add_edge("agent", END)

graph = graph.compile()
