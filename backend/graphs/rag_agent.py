from collections.abc import Callable

from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_deepseek import ChatDeepSeek

import os
import requests

chat = ChatDeepSeek(
    base_url=os.environ.get("LLM_BASE_URL", ""),
    api_key=os.environ.get("LLM_API_KEY", ""),
    model=os.environ.get("LLM_NAME", ""),
    streaming=True,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "medium",
        },
        "reasoning_format": "deepseek",
    },
)
retrieval_api_url = os.environ.get("RETRIEVAL_API_URL", "")


class ToolCallMiddleware(AgentMiddleware):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        try:
            return handler(request)
        except Exception as e:
            return ToolMessage(
                content=f"Tool error during tool execution: ({e})",
                tool_call_id=request.tool_call["id"],
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        try:
            return await handler(request)
        except Exception as e:
            return ToolMessage(
                content=f"Tool error during tool execution: ({e})",
                tool_call_id=request.tool_call["id"],
            )


def search_impl(query: str, k: int, searchname: str):
    if k >= 20:
        return {"error": "Requested too many chunks, use k less than 20"}
    request_data = {"query": query, "searchname": searchname, "k": k}
    response = requests.post(retrieval_api_url, json=request_data)
    chunks_data = response.json()["chunks"]
    chunks_data = [
        {
            "name": chunk["metadata"]["searchname"],
            "tags": chunk["metadata"]["tags"],
            "content": chunk["page_content"],
        }
        for chunk in chunks_data
    ]
    return chunks_data


@tool
def search_query(query: str, k: int = 3) -> str:
    """Search the medications database by cosine vectors similarity.

    Args:
        query: natural language search query
        k: limit number of chunks to be returned (3 by default)

    Returns:
    List of the top-k most relevant text chunks.
    """
    return search_impl(query=query, k=k, searchname=None)


@tool
def search_query_with_name(query: str, medication_name: str = "", k: int = 3) -> str:
    """Search chunks from specific medication by cosine vectors similarity. Use this only if medication_name was previously retrieved via `search_query` tool.

    Args:
        query: natural language search query
        medication_name: exact name of one specific medication (without ® or special symbols)
        k: limit number of chunks to be returned (3 by default)

    Returns:
    List of the top-k most relevant text chunks from the specified medication.
    """
    if "®" in medication_name:
        return {"error": "Medication name should not contain ®"}
    result = search_impl(query=query, k=k, searchname=medication_name)
    if not result:
        return {
            "error": "Error in a searchname or medication does not exists in database"
        }
    return result


TOOLS = [search_query, search_query_with_name]


agent = create_agent(
    model=chat,
    tools=TOOLS,
    middleware=[ToolCallMiddleware()],
)
