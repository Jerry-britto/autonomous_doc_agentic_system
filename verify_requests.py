import httpx
import sys
import os
import time

API_URL = "http://127.0.0.1:8000"

def run_test(name, payload):
    print("=" * 60)
    print(f"RUNNING TEST: {name}")
    print(f"Request: {payload['request']}")
    print("=" * 60)
    
    try:
        response = httpx.post(f"{API_URL}/agent", json=payload, timeout=30.0)
        
        if response.status_code != 200:
            print(f"Error Response: {response.text}")
            sys.exit(1)
            
        data = response.json()
        job_id = data["job_id"]
        print(f"Job queued successfully. Job ID: {job_id}")
        
        last_log_len = 0
        while True:
            status_res = httpx.get(f"{API_URL}/agent/status/{job_id}", timeout=10.0)
            if status_res.status_code != 200:
                print(f"Failed to fetch job status: {status_res.text}")
                sys.exit(1)
                
            job = status_res.json()
            
            # Print new logs
            logs = job.get("logs", [])
            if len(logs) > last_log_len:
                for log in logs[last_log_len:]:
                    print(f"  [LOG] {log}")
                last_log_len = len(logs)
                
            if job["status"] == "completed":
                print("\nJob completed successfully!")
                print(f"Generated Document Title: {job['title']}")
                print(f"Download URL: {job['download_url']}")
                
                print("\nFinal Agent Plan:")
                for idx, task in enumerate(job['plan']):
                    print(f"  {idx+1}. Task ID: {task['id']}")
                    print(f"     Description: {task['description']}")
                    print(f"     Tool: {task['assigned_tool']}")
                    print(f"     Status: {task['status']}")
                    if task.get('section_heading'):
                        print(f"     Section Heading: {task['section_heading']}")
                    snippet = task['result'][:150] + "..." if task.get('result') else ""
                    print(f"     Result Preview: {snippet}")
                    
                # Verify download
                download_url = f"{API_URL}{job['download_url']}"
                dl_response = httpx.get(download_url)
                assert dl_response.status_code == 200, f"Failed to download document from {download_url}"
                print(f"\nSUCCESS: Document successfully downloaded ({len(dl_response.content)} bytes)")
                print("=" * 60 + "\n")
                break
                
            elif job["status"] == "failed":
                print(f"\nJob failed: {job.get('error')}")
                sys.exit(1)
                
            time.sleep(2.0)
            
    except Exception as e:
        print(f"Exception during test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Test 1: Standard business request
    standard_payload = {
        "request": "Draft a comprehensive project plan for a 3-month AI marketing campaign, detailing the timeline, channels, and budget distribution."
    }
    run_test("Standard Business Request", standard_payload)
    
    # Test 2: Complex, ambiguous request
    complex_payload = {
        "request": "Write a technical design document for a scalable notification system. We have 50 million users, but I don't know what database to use. Make standard industry recommendations for us."
    }
    run_test("Complex Ambiguous Request", complex_payload)
