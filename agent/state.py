from typing import List, Dict, Any, TypedDict
from agent.store import job_store

class LoggingList(list):
    def __init__(self, job_id: str = None):
        super().__init__()
        self.job_id = job_id
        
    def append(self, item):
        print(f"[AGENT LOG] {item}", flush=True)
        super().append(item)
        if self.job_id:
            job_store.update(self.job_id, logs=list(self))

class AgentState(TypedDict):
    request: str
    document_title: str
    plan: List[Dict[str, Any]]  # Stored as dicts in state for serialization/compatibility
    current_task_id: str
    document_sections: Dict[str, str]
    final_doc_path: str
    logs: List[str]
    step_count: int
    job_id: str
