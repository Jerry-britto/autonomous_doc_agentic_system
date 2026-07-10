import os
import json
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from agent import run_agent_with_job, job_store

app = FastAPI(title="Fluid AI Document Agent")

class AgentRequest(BaseModel):
    request: str

def run_agent_background(request_text: str, job_id: str):
    try:
        run_agent_with_job(request_text, job_id)
    except Exception as e:
        pass

@app.post("/agent")
async def run_document_agent(payload: AgentRequest, background_tasks: BackgroundTasks):
    if not payload.request.strip():
        raise HTTPException(status_code=400, detail="Request text cannot be empty.")
    
    # Create unique job
    job_id = job_store.create(payload.request)
    
    # Run agent in the background
    background_tasks.add_task(run_agent_background, payload.request, job_id)
    
    return {
        "status": "queued",
        "job_id": job_id
    }

@app.get("/agent/status/{job_id}")
async def get_agent_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job

@app.get("/agent/stream/{job_id}")
async def stream_agent_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
        
    async def event_generator():
        last_plan_json = None
        last_logs_count = 0
        
        while True:
            current_job = job_store.get(job_id)
            if not current_job:
                break
                
            current_plan_json = json.dumps(current_job.get("plan", []))
            plan_changed = current_plan_json != last_plan_json
            logs_changed = len(current_job.get("logs", [])) > last_logs_count
            status = current_job.get("status")
            
            if plan_changed or logs_changed or status in ["completed", "failed"]:
                last_plan_json = current_plan_json
                last_logs_count = len(current_job.get("logs", []))
                
                payload = {
                    "status": current_job["status"],
                    "title": current_job["title"],
                    "plan": current_job["plan"],
                    "logs": current_job["logs"],
                    "download_url": current_job["download_url"],
                    "error": current_job["error"]
                }
                yield f"data: {json.dumps(payload)}\n\n"
                
            if status in ["completed", "failed"]:
                break
                
            await asyncio.sleep(0.2)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/download/{filename}")
async def download_file(filename: str):
    output_dir = os.path.join(os.getcwd(), "generated_docs")
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, filename=filename, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(os.getcwd(), "templates", "index.html")
    if not os.path.exists(index_path):
        return "Frontend HTML file not found."
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/static/style.css")
async def get_css():
    css_path = os.path.join(os.getcwd(), "static", "style.css")
    if not os.path.exists(css_path):
        raise HTTPException(status_code=404, detail="CSS file not found.")
    return FileResponse(css_path, media_type="text/css")
