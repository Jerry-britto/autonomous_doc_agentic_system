import os
import re
import json
import time
from typing import Any, List, Dict, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv()

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
    """Invokes LLM with rate limit retries, fallback to llama-3.1-8b-instant, and raw JSON parsing fallback."""
    selected_llm = primary_llm
    
    # Try primary model with retries for rate limits
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            if structured_output_model:
                model = selected_llm.with_structured_output(structured_output_model)
            else:
                model = selected_llm
            return model.invoke(prompt)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg:
                # If daily quota limit exceeded, retry won't help. We check if daily limit is mentioned.
                if "daily" in err_msg or "quota" in err_msg or attempt == max_retries:
                    msg = "Primary LLM quota exhausted or max retries reached. Switching to fallback llama-3.1-8b-instant..."
                    print(msg)
                    if logs is not None:
                        logs.append(f"System: {msg}")
                    selected_llm = fallback_llm
                    break
                else:
                    sleep_time = 3 + attempt * 2
                    msg = f"Primary LLM rate limited (attempt {attempt+1}/{max_retries}). Retrying in {sleep_time}s..."
                    print(msg)
                    if logs is not None:
                        logs.append(f"System: {msg}")
                    time.sleep(sleep_time)
            else:
                if structured_output_model:
                    print(f"Primary structured invocation failed ({e}). Trying raw JSON fallback on primary...")
                    break
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
                                "status": t.get("status", "pending") if t.get("status") in ["pending", "in_progress", "completed", "failed", "deferred"] else "pending",
                                "dependencies": t.get("dependencies", []) if isinstance(t.get("dependencies"), list) else [],
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
    """Format the plan for LLM prompts, keeping it token-efficient."""
    summary = []
    for t in plan:
        tool_info = f" [Tool: {t['assigned_tool']}]" if t['assigned_tool'] != 'none' else ""
        heading_info = f" [Heading: {t['section_heading']}]" if t.get('section_heading') else ""
        summary.append(f"- ID: {t['id']} | Description: {t['description']}{tool_info}{heading_info} | Status: {t['status']}")
        # Only show result snippets for web search tasks to save context tokens
        if t['result'] and t['assigned_tool'] == "web_search":
            snippet = t['result'][:300] + "..." if len(t['result']) > 300 else t['result']
            summary.append(f"  Research Summary: {snippet}")
    return "\n".join(summary)
