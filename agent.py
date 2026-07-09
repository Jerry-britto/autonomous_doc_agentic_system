import os
import json
import re
from typing import List, Dict, Any, TypedDict, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from doc_builder import build_academic_document

# Load environment variables
load_dotenv()

# Pydantic models for structured output
class InitialTaskPlan(BaseModel):
    id: str = Field(description="Unique short identifier, e.g., 'research_db', 'draft_intro'")
    description: str = Field(description="Detailed description of what the task should accomplish")
    assigned_tool: Literal["web_search", "draft_section", "none"] = Field(description="Tool to use: 'web_search', 'draft_section', or 'none'")
    section_heading: Optional[str] = Field(None, description="Heading of the section in the final document, e.g. '1. Introduction'")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(default="pending")
    result: str = Field(default="", description="The output result or drafted content of the task")

class InitialPlan(BaseModel):
    document_title: str = Field(description="Formal title of the document")
    tasks: List[InitialTaskPlan] = Field(description="List of initial tasks")

class UpdateTaskPlan(BaseModel):
    id: str = Field(description="Unique short identifier, e.g., 'research_db', 'draft_intro'")
    description: Optional[str] = Field(None, description="Detailed description of what the task should accomplish")
    assigned_tool: Optional[Literal["web_search", "draft_section", "none"]] = Field(None, description="Tool to use: 'web_search', 'draft_section', or 'none'")
    section_heading: Optional[str] = Field(None, description="Heading of the section in the final document, e.g. '1. Introduction'")
    status: Optional[Literal["pending", "in_progress", "completed", "failed"]] = Field(default="pending")
    result: Optional[str] = Field(default="", description="The output result or drafted content of the task")

class PlanUpdate(BaseModel):
    tasks: List[UpdateTaskPlan] = Field(description="Complete list of tasks, with updated statuses, new tasks, or modified tasks")

class LoggingList(list):
    def append(self, item):
        print(f"[AGENT LOG] {item}", flush=True)
        super().append(item)

# LangGraph State Definition
class AgentState(TypedDict):
    request: str
    document_title: str
    plan: List[Dict[str, Any]]  # Stored as dicts in state for serialization/compatibility
    current_task_id: str
    document_sections: Dict[str, str]
    final_doc_path: str
    logs: List[str]
    step_count: int

# Initialize LLM & Tools
primary_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
fallback_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
search_tool = TavilySearch()

def extract_json(text: str) -> Optional[dict]:
    # Try finding markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    # Try finding general braces
    match2 = re.search(r'(\{.*\})', text, re.DOTALL)
    if match2:
        try:
            return json.loads(match2.group(1))
        except:
            pass
    try:
        return json.loads(text.strip())
    except:
        return None

def invoke_llm(prompt: str, structured_output_model: Any = None, logs: List[str] = None) -> Any:
    """Invokes LLM with automatic fallback to llama-3.1-8b-instant if rate limited, and raw JSON parsing fallback."""
    selected_llm = primary_llm
    
    try:
        if structured_output_model:
            model = selected_llm.with_structured_output(structured_output_model)
        else:
            model = selected_llm
        return model.invoke(prompt)
    except Exception as e:
        err_msg = str(e).lower()
        if "429" in err_msg or "rate limit" in err_msg:
            msg = "Primary LLM rate limited. Switching to fallback llama-3.1-8b-instant..."
            print(msg)
            if logs is not None:
                logs.append(f"System: {msg}")
            selected_llm = fallback_llm
        else:
            if structured_output_model:
                print(f"Primary structured invocation failed ({e}). Trying raw JSON fallback on primary...")
            else:
                raise e

    try:
        if structured_output_model:
            model = selected_llm.with_structured_output(structured_output_model)
        else:
            model = selected_llm
        return model.invoke(prompt)
    except Exception as e:
        if structured_output_model:
            msg_fail = f"Structured model invocation failed: {str(e)}. Attempting raw JSON instruction & parsing..."
            print(msg_fail)
            if logs is not None:
                logs.append(f"System: {msg_fail}")
                
            schema_instruction = f"\n\nYou MUST return your output in raw JSON format matching this structure:\n"
            if structured_output_model.__name__ == "InitialPlan":
                schema_instruction += """{
  "document_title": "string title",
  "tasks": [
    {
      "id": "task_id",
      "description": "task description",
      "assigned_tool": "web_search" | "draft_section" | "none",
      "section_heading": "optional heading" or null
    }
  ]
}"""
            else:
                schema_instruction += """{
  "tasks": [
    {
      "id": "task_id",
      "description": "task description",
      "assigned_tool": "web_search" | "draft_section" | "none",
      "section_heading": "optional heading" or null,
      "status": "pending" | "in_progress" | "completed" | "failed",
      "result": "result content"
    }
  ]
}"""
            
            raw_prompt = prompt + schema_instruction
            raw_res = selected_llm.invoke(raw_prompt)
            parsed_dict = extract_json(raw_res.content)
            if parsed_dict:
                try:
                    validated = structured_output_model.model_validate(parsed_dict)
                    print("Raw JSON parsing and validation succeeded!")
                    return validated
                except Exception as val_e:
                    print("Validation of parsed JSON failed, performing fallback repair:", val_e)
                    if "tasks" in parsed_dict and isinstance(parsed_dict["tasks"], list):
                        clean_tasks = []
                        for t in parsed_dict["tasks"]:
                            clean_tasks.append({
                                "id": str(t.get("id", "task")),
                                "description": str(t.get("description", "draft")),
                                "assigned_tool": t.get("assigned_tool", "none") if t.get("assigned_tool") in ["web_search", "draft_section", "none"] else "none",
                                "section_heading": t.get("section_heading"),
                                "status": t.get("status", "pending") if t.get("status") in ["pending", "in_progress", "completed", "failed"] else "pending",
                                "result": str(t.get("result", ""))
                            })
                        parsed_dict["tasks"] = clean_tasks
                        if structured_output_model.__name__ == "InitialPlan":
                            parsed_dict["document_title"] = parsed_dict.get("document_title", "Document")
                        return structured_output_model.model_validate(parsed_dict)
            raise e
        else:
            raise e

def get_plan_summary(plan: List[Dict[str, Any]]) -> str:
    """Format the plan for LLM prompts."""
    summary = []
    for t in plan:
        tool_info = f" [Tool: {t['assigned_tool']}]" if t['assigned_tool'] != 'none' else ""
        heading_info = f" [Heading: {t['section_heading']}]" if t.get('section_heading') else ""
        summary.append(f"- ID: {t['id']} | Description: {t['description']}{tool_info}{heading_info} | Status: {t['status']}")
        if t['result']:
            snippet = t['result'][:150] + "..." if len(t['result']) > 150 else t['result']
            summary.append(f"  Result: {snippet}")
    return "\n".join(summary)

# 1. Planner Node
def planner_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    logs.append("Phase 1: Planning started. Analyzing request and forming initial plan...")
    
    prompt = f"""You are an expert document planner. A user wants to generate a professional document.
User Request: {state['request']}

Your job is to:
1. Determine a formal, clean title for the document.
2. Create an initial plan of 2 to 5 tasks to fulfill this request.

Guidelines:
- If the request is complex, ambiguous, or has missing information, include a research task (assigned_tool: 'web_search') first.
- Include tasks to draft specific sections (assigned_tool: 'draft_section' and specify the 'section_heading').
- Keep the tasks in logical order (e.g., research/outline first, then drafting sections).
- Do not create too many tasks at once; the replanner can add more tasks later based on findings.
"""
    initial_plan = invoke_llm(prompt, InitialPlan, logs)
    
    # Convert tasks to dictionaries and enforce 'pending' status
    plan_dicts = []
    for t in initial_plan.tasks:
        td = t.model_dump()
        td["status"] = "pending"
        plan_dicts.append(td)
    
    logs.append(f"Planner generated title: '{initial_plan.document_title}'")
    logs.append(f"Initial tasks: {[t['id'] for t in plan_dicts]}")
    
    return {
        "document_title": initial_plan.document_title,
        "plan": plan_dicts,
        "logs": logs
    }

# 2. Selector Node
def selector_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    plan = state["plan"]
    step_count = state.get("step_count", 0)
    
    # Guardrail: Limit maximum steps to prevent infinite planning loops
    if step_count >= 8:
        logs.append(f"Selector Guardrail: Maximum task execution limit (8) reached. Forcing compilation to prevent infinite loop.")
        return {
            "plan": plan,
            "current_task_id": "",
            "logs": logs
        }
    
    # Find the first pending task
    next_task = None
    for t in plan:
        if t["status"] == "pending":
            next_task = t
            break
            
    if next_task:
        # Mark as in_progress
        for t in plan:
            if t["id"] == next_task["id"]:
                t["status"] = "in_progress"
                
        current_task_id = next_task["id"]
        logs.append(f"Selector: Selected task '{current_task_id}' for execution.")
    else:
        current_task_id = ""
        logs.append("Selector: No pending tasks remaining. Document generation phase starting.")
        
    return {
        "plan": plan,
        "current_task_id": current_task_id,
        "logs": logs
    }

# 3. Executor Node
def executor_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    plan = state["plan"]
    current_task_id = state["current_task_id"]
    
    # Find current task details
    task = None
    for t in plan:
        if t["id"] == current_task_id:
            task = t
            break
            
    if not task:
        logs.append(f"Executor Error: Task '{current_task_id}' not found.")
        return {}
        
    logs.append(f"Executor: Running task '{current_task_id}' [Tool: {task['assigned_tool']}]...")
    
    result_content = ""
    
    # Execute tool
    if task["assigned_tool"] == "web_search":
        try:
            # Step 1: Run Tavily search
            search_query = task["description"]
            logs.append(f"Executor: Searching web for '{search_query}'...")
            search_res = search_tool.invoke(search_query)
            
            # Step 2: Use LLM to summarize search results
            logs.append(f"Executor: Summarizing search findings...")
            summarize_prompt = f"""You are a professional researcher. Summarize the following web search results to answer the task: "{task['description']}".
Search Results: {search_res}

Provide a detailed, structured, and informative summary of findings, data, and recommendations. Use markdown format.
"""
            summary_res = invoke_llm(summarize_prompt, logs=logs)
            result_content = summary_res.content
            logs.append(f"Executor: Research compiled successfully (size: {len(result_content)} chars).")
        except Exception as e:
            result_content = f"Search error occurred: {str(e)}"
            logs.append(f"Executor Error during search: {str(e)}")
            
    elif task["assigned_tool"] == "draft_section":
        # Draft section content
        plan_summary = get_plan_summary(plan)
        draft_prompt = f"""You are a professional academic writer. Draft the section: "{task['section_heading']}".
Task Description: {task['description']}
Original User Request: {state['request']}

Here is the plan and research context we have gathered so far:
{plan_summary}

Draft this section in clean Markdown.
Guidelines:
- Include headings, subheadings (### or ####), bullet points, and tables where appropriate.
- For tables, format them using standard markdown syntax, e.g.:
  | Header 1 | Header 2 |
  |----------|----------|
  | Cell 1   | Cell 2   |
- Make it extremely detailed, authoritative, and structured.
- Adhere strictly to the requested information and findings.
"""
        logs.append(f"Executor: Drafting section '{task['section_heading']}'...")
        draft_res = invoke_llm(draft_prompt, logs=logs)
        result_content = draft_res.content
        logs.append(f"Executor: Section drafted successfully (size: {len(result_content)} chars).")
        
    else:
        # Generic LLM execution
        plan_summary = get_plan_summary(plan)
        generic_prompt = f"""You are an AI assistant executing a step in a document generation plan.
Task to execute: {task['description']}
Original Request: {state['request']}
Current Plan/Context:
{plan_summary}

Execute the task and return the findings/content.
"""
        logs.append(f"Executor: Executing generic task...")
        gen_res = invoke_llm(generic_prompt, logs=logs)
        result_content = gen_res.content
        logs.append(f"Executor: Generic task finished.")
        
    # Update the task result in our local copy of plan
    for t in plan:
        if t["id"] == current_task_id:
            t["result"] = result_content
            
    step_count = state.get("step_count", 0) + 1
    return {
        "plan": plan,
        "logs": logs,
        "step_count": step_count
    }

# 4. Replanner Node
def replanner_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    plan = state["plan"]
    current_task_id = state["current_task_id"]
    document_sections = state.get("document_sections", {})
    
    # Find current task details
    current_task = None
    for t in plan:
        if t["id"] == current_task_id:
            current_task = t
            break
            
    if not current_task:
        return {}
        
    # If the task drafted a section, add it to our document_sections map
    if current_task["assigned_tool"] == "draft_section" and current_task["section_heading"]:
        document_sections[current_task["section_heading"]] = current_task["result"]
        logs.append(f"Replanner: Added section '{current_task['section_heading']}' to document content.")
        
    logs.append(f"Replanner: Re-evaluating plan after task '{current_task_id}' completion...")
    
    plan_summary = get_plan_summary(plan)
    
    prompt = f"""You are an expert re-planner. Review the execution of the active task and update the document plan.
Original User Request: {state['request']}
Active Task ID: {current_task_id}
Active Task Description: {current_task['description']}
Active Task Result: {current_task['result'][:1000]}... (truncated)

Current Plan & Status:
{plan_summary}

Your job is to update the plan. You must return the COMPLETE list of tasks.
Guidelines:
- Mark the active task '{current_task_id}' as 'completed' (or 'failed' if it failed).
- Add new tasks if you need to draft more sections, execute other sub-steps, or perform further research based on the findings.
- Keep pending tasks if they are still relevant.
- Update pending tasks if their descriptions should change based on new findings.
- If all sections have been drafted to fulfill the user request, do not add any new tasks.
"""
    plan_update = invoke_llm(prompt, PlanUpdate, logs)
    
    # Convert LLM task models to dicts
    updated_plan_dicts = [t.model_dump() for t in plan_update.tasks]
    
    # Create lookup map for existing tasks from the previous state
    old_tasks = {t["id"]: t for t in plan}
    
    # Re-apply status and results from previous state to prevent LLM hallucinations
    for t in updated_plan_dicts:
        old_t = old_tasks.get(t["id"])
        if old_t:
            # Task existed in previous state: restore omitted fields if missing
            if not t.get("description"):
                t["description"] = old_t.get("description", "")
            if not t.get("assigned_tool"):
                t["assigned_tool"] = old_t.get("assigned_tool", "none")
            if not t.get("section_heading"):
                t["section_heading"] = old_t.get("section_heading")
                
            if t["id"] == current_task_id:
                # Update current active task status
                if "error occurred" in str(old_t.get("result", "")).lower() or t.get("status") == "failed":
                    t["status"] = "failed"
                else:
                    t["status"] = "completed"
                t["result"] = old_t.get("result", "")
            else:
                # Preserve original status and results of other existing tasks
                t["status"] = old_t.get("status", "pending")
                t["result"] = old_t.get("result", "")
        else:
            # Newly added task: must start as pending
            t["status"] = "pending"
            t["result"] = ""
            # Set default values if missing
            if not t.get("description"):
                t["description"] = "Drafting new section content"
            if not t.get("assigned_tool"):
                t["assigned_tool"] = "none"
            
    # Trace differences
    old_task_ids = list(old_tasks.keys())
    new_task_ids = [t["id"] for t in updated_plan_dicts]
    added_tasks = [tid for tid in new_task_ids if tid not in old_task_ids]
    
    if added_tasks:
        logs.append(f"Replanner: Dynamically added tasks to plan: {added_tasks}")
    else:
        logs.append("Replanner: Plan checked. No new tasks added.")
        
    return {
        "plan": updated_plan_dicts,
        "document_sections": document_sections,
        "logs": logs
    }

# 5. Generator Node
def generator_node(state: AgentState) -> Dict[str, Any]:
    logs = state.get("logs", [])
    logs.append("Phase 3: Compiling document sections into Microsoft Word format...")
    
    title = state.get("document_title", "Business Report")
    plan = state["plan"]
    document_sections = state["document_sections"]
    
    # Gather sections in the logical order they were completed/drafted
    sections_to_build = []
    # Keep track of what we added to prevent duplicates
    added_headings = set()
    
    # Add sections based on completed tasks
    for t in plan:
        if t["assigned_tool"] == "draft_section" and t["section_heading"] and t["status"] == "completed":
            heading = t["section_heading"]
            content = t["result"]
            if heading not in added_headings:
                sections_to_build.append({
                    "heading": heading,
                    "content": content
                })
                added_headings.add(heading)
                
    # Fallback to document_sections dict for anything missing
    for heading, content in document_sections.items():
        if heading not in added_headings:
            sections_to_build.append({
                "heading": heading,
                "content": content
            })
            added_headings.add(heading)
            
    # Generate clean filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip()
    safe_title = safe_title.replace(" ", "_")
    filename = f"{safe_title}.docx"
    
    output_dir = os.path.join(os.getcwd(), "generated_docs")
    output_path = os.path.join(output_dir, filename)
    
    try:
        build_academic_document(title, sections_to_build, output_path)
        logs.append(f"Generator: Word document generated successfully at '{output_path}'.")
    except Exception as e:
        output_path = f"Error generating file: {str(e)}"
        logs.append(f"Generator Error: {str(e)}")
        
    return {
        "final_doc_path": output_path,
        "logs": logs
    }

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
        "step_count": 0
    }
    
    final_state = graph.invoke(initial_state)
    return final_state
