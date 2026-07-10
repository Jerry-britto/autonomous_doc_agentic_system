import os
import json
import uuid
from typing import Optional

# Singleton JobStore for asynchronous API tracking
class JobStore:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JobStore, cls).__new__(cls)
            cls._instance.filepath = os.path.join(os.getcwd(), "jobs.json")
            cls._instance.jobs = {}
            cls._instance.load()
        return cls._instance
        
    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.jobs = json.load(f)
            except Exception as e:
                print(f"Error loading jobs from disk: {e}")
                self.jobs = {}
                
    def save(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.jobs, f, indent=2)
        except Exception as e:
            print(f"Error saving jobs to disk: {e}")
            
    def get(self, job_id: str) -> Optional[dict]:
        return self.jobs.get(job_id)
        
    def update(self, job_id: str, **kwargs):
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)
            self.save()
            
    def create(self, request_text: str) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "status": "planning",
            "title": "Document Generator",
            "plan": [],
            "logs": ["Job created. Starting autonomous planning agent..."],
            "download_url": None,
            "error": None
        }
        self.save()
        return job_id

job_store = JobStore()
