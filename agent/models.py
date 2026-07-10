from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class InitialTaskPlan(BaseModel):
    id: str = Field(description="Unique short identifier, e.g., 'research_db', 'draft_intro'")
    description: str = Field(description="Detailed description of what the task should accomplish")
    assigned_tool: Literal["web_search", "draft_section", "none"] = Field(description="Tool to use: 'web_search', 'draft_section', or 'none'")
    section_heading: Optional[str] = Field(None, description="Heading of the section in the final document, e.g. '1. Introduction'")
    status: Literal["pending", "in_progress", "completed", "failed", "deferred"] = Field(default="pending")
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks that must be completed before this task can start")
    result: str = Field(default="", description="The output result or drafted content of the task")

class InitialPlan(BaseModel):
    document_title: str = Field(description="Formal title of the document")
    tasks: List[InitialTaskPlan] = Field(description="List of initial tasks")

class UpdateTaskPlan(BaseModel):
    id: str = Field(description="Unique short identifier, e.g., 'research_db', 'draft_intro'")
    description: Optional[str] = Field(None, description="Detailed description of what the task should accomplish")
    assigned_tool: Optional[Literal["web_search", "draft_section", "none"]] = Field(None, description="Tool to use: 'web_search', 'draft_section', or 'none'")
    section_heading: Optional[str] = Field(None, description="Heading of the section in the final document, e.g. '1. Introduction'")
    status: Optional[Literal["pending", "in_progress", "completed", "failed", "deferred"]] = Field(default="pending")
    dependencies: Optional[List[str]] = Field(default_factory=list, description="IDs of tasks that must be completed before this task can start")
    result: Optional[str] = Field(default="", description="The output result or drafted content of the task")

class PlanUpdate(BaseModel):
    tasks: List[UpdateTaskPlan] = Field(description="Complete list of tasks, with updated statuses, new tasks, or modified tasks")
