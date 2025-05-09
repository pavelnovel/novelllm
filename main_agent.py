from typing import Annotated, List, TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import operator
import json
import logging

# Import your tools
from tools import query_knowledge_base

# Define a simple state with proper typing
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# Create the LLM and bind tools once
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)
tools = [query_knowledge_base]
LLM_WITH_TOOLS = llm.bind_tools(tools)

# Create tool registry for easy lookup
TOOL_REGISTRY = {tool.name: tool.run for tool in tools}

# System message for the agent
SYSTEM_MESSAGE = """You are a helpful assistant that can search through documents. 
Use the query_knowledge_base tool to search for information. 
After getting the search results, return the COMPLETE response including both the answer and all sources.
DO NOT summarize or modify the response in any way.
IMPORTANT: Only make one tool call at a time.
IMPORTANT: Always preserve and include ALL source information from the tool's response."""

# Agent node function
def agent(state: AgentState) -> Dict[str, Any]:
    """Agent node that generates responses."""
    msgs = state["messages"]
    
    # Insert system message if not present
    if not any(isinstance(m, SystemMessage) for m in msgs):
        msgs = [SystemMessage(content=SYSTEM_MESSAGE)] + msgs
    
    # Generate a response using the LLM with tools
    try:
        result = LLM_WITH_TOOLS.invoke(msgs)
        logging.debug(f"Generated response type: {type(result).__name__}")
        if hasattr(result, "tool_calls") and result.tool_calls:
            logging.debug(f"Response contains {len(result.tool_calls)} tool calls")
        return {"messages": [result]}
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        return {"messages": [AIMessage(content=f"Error: {str(e)}")]}

# Tool execution node
def run_tools(state: AgentState) -> Dict[str, Any]:
    """Execute tools called by the agent."""
    last = state["messages"][-1]
    tool_messages = []
    
    # Get tool_calls, handling both object and dict formats
    tool_calls = []
    if hasattr(last, "tool_calls"):
        tool_calls = last.tool_calls
    
    for call in tool_calls:
        try:
            # Handle both object and dict styles for the tool call
            if isinstance(call, dict):
                tool_name = call.get("name")
                tool_id = call.get("id")
                # Get arguments from different possible locations
                if "arguments" in call:
                    args_str = call["arguments"]
                elif "args" in call:
                    args_str = call["args"]
                else:
                    args_str = "{}"
            else:
                # Object style
                tool_name = call.name if hasattr(call, "name") else None
                tool_id = call.id if hasattr(call, "id") else None
                args_str = call.arguments if hasattr(call, "arguments") else call.args if hasattr(call, "args") else "{}"
            
            # Parse arguments - handle both string and dict formats
            if isinstance(args_str, str):
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {"topic": args_str}  # Fallback for query_knowledge_base
            else:
                args = args_str
                
            print(f"Executing tool {tool_name} with args: {args}")
            
            # Special case for query_knowledge_base which expects a single string
            if tool_name == "query_knowledge_base":
                if isinstance(args, dict) and "topic" in args:
                    result = TOOL_REGISTRY[tool_name](args["topic"])
                else:
                    result = TOOL_REGISTRY[tool_name](str(args))
            else:
                # Generic tool execution with kwargs
                result = TOOL_REGISTRY[tool_name](**args)
            
            # Create tool message with the result
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_id)
            )
        except Exception as e:
            print(f"Error executing tool: {str(e)}")
            # Get tool_id even if we encountered an error earlier
            tool_id = None
            if isinstance(call, dict):
                tool_id = call.get("id")
            else:
                tool_id = call.id if hasattr(call, "id") else None
                
            if tool_id:
                tool_messages.append(
                    ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_id)
                )
    
    return {"messages": tool_messages}

# Simple routing function that handles both dict and object styles
def route(state):
    last = state["messages"][-1]
    
    # Check if the last message has tool calls
    has_tool_calls = False
    
    if isinstance(last, AIMessage):
        if hasattr(last, "tool_calls") and last.tool_calls:
            has_tool_calls = True
    
    return "tools" if has_tool_calls else END

# Create a graph
def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent)
    workflow.add_node("tools", run_tools)
    
    # Add conditional edges with the new route function
    workflow.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Set the entry point
    workflow.set_entry_point("agent")
    
    return workflow.compile()

# Initialize the graph once
graph = build_graph()

def process_query(query: str) -> str:
    """Process a single query through the agent."""
    state = {"messages": [HumanMessage(content=query)]}
    result = graph.invoke(state)
    
    # Extract the final response
    final_message = result["messages"][-1]
    if isinstance(final_message, AIMessage):
        if "You wrote about" in final_message.content or "Marketing is" in final_message.content:
            tool_message = result["messages"][-2]
            if isinstance(tool_message, ToolMessage):
                return tool_message.content
        return final_message.content
    elif isinstance(final_message, ToolMessage):
        return final_message.content
    else:
        return "No response generated."

if __name__ == "__main__":
    # Basic logging setup
    logging.basicConfig(level=logging.INFO)
    
    # Take input from the user
    user_input = input("Ask something: ")
    response = process_query(user_input)
    print("\nResponse:")
    print("---------")
    print(response)