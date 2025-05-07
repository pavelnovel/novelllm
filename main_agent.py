from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tools import query_knowledge_base
import operator
import json

# Define the state type
class AgentState(TypedDict):
    messages: Annotated[Sequence, operator.add]
    next: str

# Create tools list
tools = [query_knowledge_base]

# Initialize the language model with tool calling enabled
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,
    model_kwargs={
        "tools": [{
            "type": "function",
            "function": {
                "name": "query_knowledge_base",
                "description": "Run a semantic query against the embedded corpus.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The topic to search for in the knowledge base"
                        }
                    },
                    "required": ["topic"]
                }
            }
        }]
    }
)

def manage_message_history(messages, max_messages=5):
    """Manage message history to prevent exceeding token limits."""
    if len(messages) <= max_messages:
        return messages
    
    # Always keep the system message if it exists
    system_message = next((msg for msg in messages if msg.get("role") == "system"), None)
    
    # Keep the most recent messages
    recent_messages = messages[-max_messages:]
    
    # If we had a system message and it's not in the recent messages, add it back
    if system_message and system_message not in recent_messages:
        return [system_message] + recent_messages
    
    return recent_messages

def should_continue(state: AgentState) -> str:
    """Determine the next node based on the state."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message is from the assistant and has no tool calls, we're done
    if last_message["role"] == "assistant" and not last_message.get("tool_calls"):
        return END
    
    # If the last message is a tool response, go back to agent
    if last_message["role"] == "tool":
        return "agent"
    
    # If the last message is from the assistant and has tool calls, go to tools
    if last_message["role"] == "assistant" and last_message.get("tool_calls"):
        return "tools"
    
    # Default to ending
    return END

def agent_node(state: AgentState):
    """Agent node that decides what to do next."""
    messages = state["messages"]
    
    # Add system message if it's the first message
    if len(messages) == 1 and messages[0].get("role") != "system":
        messages = [
            {"role": "system", "content": "You are a helpful assistant that can search through documents. Use the query_knowledge_base tool to search for information. After getting the search results, provide a final answer based on those results and DO NOT make additional tool calls. IMPORTANT: Only make one tool call at a time."}
        ] + messages
    
    # Manage message history to prevent exceeding token limits
    messages = manage_message_history(messages)
    
    print(f"\nDebug - Message count: {len(messages)}")
    print("\n🤖 Assistant thinking...")
    
    try:
        response = llm.invoke(messages)
        print(f"Debug - Response content: {response.content}")
        print(f"Debug - Has tool calls: {hasattr(response, 'tool_calls')}")
        if hasattr(response, 'tool_calls'):
            print(f"Debug - Tool calls: {response.tool_calls}")
        
        # Check if the response contains tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"Debug - Processing tool call")
            tool_call = response.tool_calls[0]
            
            # Create the tool call message in the correct format
            tool_call_message = {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [{
                    "id": tool_call.get("id", f"call_{hash(str(tool_call))}"),  # Generate ID if not present
                    "type": "function",
                    "function": {
                        "name": tool_call.get("name", tool_call.get("function", {}).get("name")),
                        "arguments": json.dumps(tool_call.get("args", tool_call.get("function", {}).get("arguments", {})))
                    }
                }]
            }
            
            return {
                "messages": messages + [tool_call_message]
            }
        
        # If no valid tool calls, just return the response
        print("Debug - No tool calls, returning final response")
        return {"messages": messages + [{"role": "assistant", "content": response.content}]}
    
    except Exception as e:
        print(f"Error in agent_node: {str(e)}")
        return {
            "messages": messages + [
                {"role": "assistant", "content": "I apologize, but I encountered an error processing your request. Could you try asking in a different way?"}
            ]
        }

def tools_node(state: AgentState):
    """Tools node that executes the selected tool."""
    messages = state["messages"]
    last_message = messages[-1]
    
    try:
        if "tool_calls" not in last_message:
            print("Debug - No tool calls found in last message")
            return {"messages": messages}
        
        tool_call = last_message["tool_calls"][0]
        tool_name = tool_call["function"]["name"]
        tool_args = tool_call["function"]["arguments"]
        tool_call_id = tool_call["id"]
        
        print(f"\n🔧 Executing tool: {tool_name}")
        print(f"Debug - Tool args: {tool_args}")
        
        # Find and execute the matching tool
        result = "Tool execution failed."
        for tool in tools:
            if tool.name == tool_name:
                try:
                    # Parse arguments if they're in JSON format
                    if isinstance(tool_args, str):
                        args_dict = json.loads(tool_args)
                    else:
                        args_dict = tool_args
                    
                    # Ensure we have a topic parameter
                    if isinstance(args_dict, dict) and "topic" in args_dict:
                        result = tool.invoke(args_dict)
                    else:
                        result = tool.invoke({"topic": str(args_dict)})
                except json.JSONDecodeError:
                    # Fallback if parsing fails
                    result = tool.invoke({"topic": str(tool_args)})
                break
        
        # Limit the size of the result if it's too large
        result_str = str(result)
        if len(result_str) > 1000:  # Truncate large results
            result_str = result_str[:1000] + "... [Result truncated due to size]"
        
        print(f"Debug - Tool result length: {len(result_str)}")
        
        # Add the result to messages with proper tool response format
        return {
            "messages": messages + [{
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_str
            }]
        }
    
    except Exception as e:
        print(f"Error in tools_node: {str(e)}")
        if "tool_calls" in last_message and len(last_message["tool_calls"]) > 0:
            tool_call_id = last_message["tool_calls"][0]["id"]
            return {
                "messages": messages + [{
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Error executing tool: {str(e)}"
                }]
            }
        else:
            return {
                "messages": messages + [{
                    "role": "assistant",
                    "content": "I encountered an error and couldn't process your request."
                }]
            }

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_conditional_edges(
    "tools",
    should_continue,
    {
        "agent": "agent",
        END: END
    }
)

# Set entry point
workflow.set_entry_point("agent")

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    # Initialize the state
    state = {
        "messages": [{"role": "user", "content": input("Ask something: ")}],
        "next": "agent"
    }
    
    # Run the graph
    for output in app.stream(state):
        if "messages" in output:
            last_message = output["messages"][-1]
            if last_message["role"] == "assistant" and not last_message.get("tool_calls"):
                print("\n🧠 Answer:\n", last_message["content"])
