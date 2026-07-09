import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from agent import run_agent

app = FastAPI(title="Fluid AI Document Agent")

class AgentRequest(BaseModel):
    request: str

@app.post("/agent")
async def run_document_agent(payload: AgentRequest):
    if not payload.request.strip():
        raise HTTPException(status_code=400, detail="Request text cannot be empty.")
    try:
        # Run agent synchronous
        result = run_agent(payload.request)
        
        path = result.get("final_doc_path", "")
        filename = os.path.basename(path) if path and os.path.exists(path) else ""
        
        return {
            "status": "success" if filename else "failed",
            "message": f"Document generated: {result.get('document_title')}" if filename else "Failed to generate document.",
            "title": result.get("document_title", ""),
            "plan": result.get("plan", []),
            "logs": result.get("logs", []),
            "download_url": f"/download/{filename}" if filename else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
