from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState, LoggingList
from agent.store import job_store
from agent.nodes import (
    planner_node,
    selector_node,
    executor_node,
    replanner_node,
    generator_node
)

# Router logic
def selector_router(state: AgentState) -> Literal["executor_node", "generator_node"]:
    if state["current_task_id"]:
        return "executor_node"
    return "generator_node"

# LangGraph Construction
def create_agent_graph():
    builder = StateGraph(AgentState)
    
    # Add Nodes
    builder.add_node("planner_node", planner_node)
    builder.add_node("selector_node", selector_node)
    builder.add_node("executor_node", executor_node)
    builder.add_node("replanner_node", replanner_node)
    builder.add_node("generator_node", generator_node)
    
    # Add Edges
    builder.add_edge(START, "planner_node")
    builder.add_edge("planner_node", "selector_node")
    
    # Conditional Routing from Selector
    builder.add_conditional_edges(
        "selector_node",
        selector_router,
        {
            "executor_node": "executor_node",
            "generator_node": "generator_node"
        }
    )
    
    builder.add_edge("executor_node", "replanner_node")
    builder.add_edge("replanner_node", "selector_node")
    builder.add_edge("generator_node", END)
    
    return builder.compile()

# Test runner helper
def run_agent(request_text: str) -> Dict[str, Any]:
    graph = create_agent_graph()
    initial_state = {
        "request": request_text,
        "document_title": "",
        "plan": [],
        "current_task_id": "",
        "document_sections": {},
        "final_doc_path": "",
        "logs": LoggingList(),
        "step_count": 0,
        "job_id": ""
    }
    
    final_state = graph.invoke(initial_state)
    return final_state

# Background execution runner helper using JobStore
def run_agent_with_job(request_text: str, job_id: str) -> Dict[str, Any]:
    graph = create_agent_graph()
    initial_state = {
        "request": request_text,
        "document_title": "",
        "plan": [],
        "current_task_id": "",
        "document_sections": {},
        "final_doc_path": "",
        "logs": LoggingList(job_id),
        "step_count": 0,
        "job_id": job_id
    }
    try:
        final_state = graph.invoke(initial_state)
        return final_state
    except Exception as e:
        msg = f"Agent execution failed with error: {str(e)}"
        print(f"[AGENT ERROR] {msg}", flush=True)
        job_store.update(job_id, status="failed", error=str(e))
        # Log error in job
        job = job_store.get(job_id)
        if job:
            job["logs"].append(msg)
            job_store.update(job_id, logs=job["logs"])
        raise e
