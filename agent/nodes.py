import os
import re
import time
from typing import Dict, Any

from agent.state import AgentState
from agent.models import InitialPlan, PlanUpdate
from agent.store import job_store
from agent.llm import invoke_llm, get_plan_summary, search_tool
from agent.doc_builder import build_academic_document

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
- DOCUMENT STRUCTURE: Every plan MUST include a task to draft the introduction or overview as the very first document section (e.g., "1. Introduction" or "1. Overview").
- TOOLS & HEADINGS: Any task that drafts content for a section of the final document MUST use `assigned_tool: 'draft_section'` and specify its corresponding `section_heading` (e.g., "1. Introduction"). Tasks that are just for research, outline, review, or administrative operations should use `assigned_tool: 'none'` and have a null `section_heading`.
- Keep the tasks in logical order (e.g., research/outline first, then drafting sections).
- Do not create too many tasks at once; the replanner can add more tasks later based on findings.
- DEPENDENCIES: For each task, specify its 'dependencies' as a list of other task IDs that must be completed before this task can start (e.g., drafting tasks should depend on the corresponding research tasks). If there are no prerequisites, leave it empty.
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
    
    # Update JobStore if job_id is present
    job_id = state.get("job_id")
    if job_id:
        job_store.update(job_id, status="running", title=initial_plan.document_title, plan=plan_dicts)
        
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
    if step_count >= 12:
        logs.append(f"Selector Guardrail: Maximum task execution limit (12) reached. Forcing compilation to prevent infinite loop.")
        # Mark all pending or in-progress tasks as deferred
        for t in plan:
            if t["status"] in ["pending", "in_progress"]:
                t["status"] = "deferred"
                
        # Update JobStore if job_id is present
        job_id = state.get("job_id")
        if job_id:
            job_store.update(job_id, plan=list(plan))
            
        return {
            "plan": plan,
            "current_task_id": "",
            "logs": logs
        }
    
    # Find the first pending task whose dependencies are all completed
    next_task = None
    for t in plan:
        if t["status"] == "pending":
            # Check if all dependency tasks are completed
            deps = t.get("dependencies", [])
            if not isinstance(deps, list):
                deps = []
            
            all_dependencies_met = True
            for dep_id in deps:
                dep_task = next((x for x in plan if x["id"] == dep_id), None)
                if dep_task and dep_task["status"] != "completed":
                    all_dependencies_met = False
                    break
                    
            if all_dependencies_met:
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
        pending_remaining = [t["id"] for t in plan if t["status"] == "pending"]
        if pending_remaining:
            logs.append(f"Selector: Remaining pending tasks {pending_remaining} have unsatisfied dependencies. Deferring execution.")
            for t in plan:
                if t["status"] in ["pending", "in_progress"]:
                    t["status"] = "deferred"
        else:
            logs.append("Selector: No pending tasks remaining. Document generation phase starting.")
        
    # Update JobStore if job_id is present
    job_id = state.get("job_id")
    if job_id:
        job_store.update(job_id, plan=list(plan))
        
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
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Step 1: Run Tavily search
                search_query = task["description"]
                logs.append(f"Executor: Searching web for '{search_query}' (attempt {attempt + 1}/{max_retries})...")
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
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logs.append(f"Executor: Search attempt {attempt + 1} failed: {str(e)}. Retrying in 4s...")
                    time.sleep(4.0)
                else:
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
    
    # Update JobStore if job_id is present
    job_id = state.get("job_id")
    if job_id:
        job_store.update(job_id, plan=list(plan))
        
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
    
    # Format a highly concise snippet of the active task result to save tokens
    if current_task["assigned_tool"] == "web_search":
        active_result_snippet = current_task['result'][:500] + "..." if len(current_task['result']) > 500 else current_task['result']
    else:
        active_result_snippet = current_task['result'][:150] + "..." if len(current_task['result']) > 150 else current_task['result']
        
    prompt = f"""You are an expert re-planner. Review the execution of the active task and update the document plan.
Original User Request: {state['request']}
Active Task ID: {current_task_id}
Active Task Description: {current_task['description']}
Active Task Result: {active_result_snippet}

Current Plan & Status:
{plan_summary}

Your job is to update the plan. You must return the COMPLETE list of tasks.
Guidelines:
- Mark the active task '{current_task_id}' as 'completed' (or 'failed' if it failed).
- Keep pending tasks if they are still relevant.
- Update pending tasks if their descriptions should change based on new findings.
- If all sections have been drafted to fulfill the user request, do not add any new tasks.
- DOCUMENT STRUCTURE: Every plan MUST include a task to draft the introduction or overview as the very first document section (e.g., "1. Introduction" or "1. Overview").
- TASK MINIMIZATION: Only add new tasks if they are absolutely critical to resolving an essential information/content gap. Do not add micro-tasks or redundant sub-tasks.
- TOOLS & HEADINGS: Any task that drafts content for a section of the final document MUST use `assigned_tool: 'draft_section'` and specify its corresponding `section_heading` (e.g., "2. Campaign Timeline"). Tasks that are just for research, outline, review, or administrative operations should use `assigned_tool: 'none'` and have a null `section_heading`.
- DEPENDENCIES: For any task you add, specify its 'dependencies' as a list of task IDs that must be completed before this task can start (e.g., a section drafting task depends on its corresponding research task).
"""
    plan_update = invoke_llm(prompt, PlanUpdate, logs)
    
    # Create lookup map for existing tasks from the previous state
    old_tasks = {t["id"]: t for t in plan}
    
    # Convert LLM task models to dicts
    updated_plan_dicts = [t.model_dump() for t in plan_update.tasks]
    
    # Heuristic auto-correction for drafting tasks to make sure they are not run with tool 'none'
    for t in updated_plan_dicts:
        if ("draft" in t["id"].lower() or "write" in t["id"].lower()) and t.get("assigned_tool") == "none":
            if not any(k in t["id"].lower() for k in ["review", "publish", "finalize", "format", "compil"]):
                t["assigned_tool"] = "draft_section"
                if not t.get("section_heading"):
                    derived_heading = t["id"].replace("draft_section_", "").replace("draft_", "").replace("write_", "").replace("_", " ").title()
                    t["section_heading"] = derived_heading
                    
    # Filter out out-of-scope development/operational tasks added by LLM (which cause planning loops)
    filtered_updated_tasks = []
    out_of_scope_keywords = ["testing", "deployment", "monitoring", "maintenance", "upgrade", "retirement", "implementation", "development", "deploy", "monitor", "maintain", "retire", "code"]
    for t in updated_plan_dicts:
        is_new = t["id"] not in old_tasks
        if is_new and any(kw in t["id"].lower() for kw in out_of_scope_keywords):
            continue
        filtered_updated_tasks.append(t)
    updated_plan_dicts = filtered_updated_tasks
    
    new_task_ids = {t["id"] for t in updated_plan_dicts}
    
    # Preserve any completed, failed, or deferred tasks that the LLM omitted
    preserved_old_tasks = []
    for old_id, old_t in old_tasks.items():
        if old_id not in new_task_ids:
            if old_t.get("status") in ["completed", "failed", "deferred"]:
                preserved_old_tasks.append(old_t)
    
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
            # Preserve dependencies
            if "dependencies" not in t or not t["dependencies"]:
                t["dependencies"] = old_t.get("dependencies", [])
                
            if t["id"] == current_task_id:
                # Update current active task status
                if "error occurred" in str(old_t.get("result", "")).lower() or t.get("status") == "failed":
                    t["status"] = "failed"
                else:
                    t["status"] = "completed"
                t["result"] = old_t.get("result", "")
            else:
                # Preserve original status and results of other existing tasks
                if old_t.get("status") == "deferred":
                    t["status"] = "deferred"
                else:
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
            if "dependencies" not in t or not t["dependencies"]:
                t["dependencies"] = []
            
    # Trace differences
    old_task_ids = list(old_tasks.keys())
    new_task_ids_list = [t["id"] for t in updated_plan_dicts]
    added_tasks = [tid for tid in new_task_ids_list if tid not in old_task_ids]
    
    if added_tasks:
        logs.append(f"Replanner: Dynamically added tasks to plan: {added_tasks}")
    else:
        logs.append("Replanner: Plan checked. No new tasks added.")
        
    final_plan_list = preserved_old_tasks + updated_plan_dicts
        
    # Update JobStore if job_id is present
    job_id = state.get("job_id")
    if job_id:
        job_store.update(job_id, plan=list(final_plan_list))
        
    return {
        "plan": final_plan_list,
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
    # Gather sections, merging content for tasks sharing the same heading
    sections_map = {} # heading -> list of content strings
    
    # Add sections based on completed tasks
    for t in plan:
        if t["assigned_tool"] == "draft_section" and t["section_heading"] and t["status"] == "completed":
            heading = t["section_heading"]
            content = t["result"]
            if heading not in sections_map:
                sections_map[heading] = []
            sections_map[heading].append(content)
            
    # Fallback to document_sections dict for anything missing
    for heading, content in document_sections.items():
        if heading not in sections_map:
            sections_map[heading] = [content]
            
    sections_to_build = []
    for heading, contents in sections_map.items():
        sections_to_build.append({
            "heading": heading,
            "content": "\n\n".join(contents)
        })
            
    # Sort sections numerically by heading prefix (e.g., "1. Introduction" -> [1], "3.1.2 Database" -> [3, 1, 2])
    def parse_heading_numbers(heading_str: str) -> list:
        match = re.match(r'^([\d\.]+)', heading_str.strip())
        if match:
            prefix = match.group(1)
            parts = []
            for p in prefix.split('.'):
                if p.strip():
                    try:
                        parts.append(int(p.strip()))
                    except ValueError:
                        pass
            if parts:
                return parts
        return [999]

    sections_to_build.sort(key=lambda s: parse_heading_numbers(s["heading"]))

    # Generate clean filename
    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip()
    safe_title = safe_title.replace(" ", "_")
    filename = f"{safe_title}.docx"
    
    output_dir = os.path.join(os.getcwd(), "generated_docs")
    output_path = os.path.join(output_dir, filename)
    
    try:
        build_academic_document(title, sections_to_build, output_path)
        logs.append(f"Generator: Word document generated successfully at '{output_path}'.")
        job_id = state.get("job_id")
        if job_id:
            job_store.update(job_id, status="completed", download_url=f"/download/{filename}")
    except Exception as e:
        output_path = f"Error generating file: {str(e)}"
        logs.append(f"Generator Error: {str(e)}")
        job_id = state.get("job_id")
        if job_id:
            job_store.update(job_id, status="failed", error=str(e))
        
    return {
        "final_doc_path": output_path,
        "logs": logs
    }
