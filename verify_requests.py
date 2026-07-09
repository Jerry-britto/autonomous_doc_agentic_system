import httpx
import sys
import os

API_URL = "http://127.0.0.1:8000"

def run_test(name, payload):
    print("=" * 60)
    print(f"RUNNING TEST: {name}")
    print(f"Request: {payload['request']}")
    print("=" * 60)
    
    try:
        # Increase timeout because agent runs multi-step planning loops
        response = httpx.post(f"{API_URL}/agent", json=payload, timeout=600.0)
        
        if response.status_code != 200:
            print(f"Error Response: {response.text}")
            sys.exit(1)
            
        data = response.json()
        print(f"Status: {data['status']}")
        print(f"Generated Document Title: {data['title']}")
        print(f"Download URL: {data['download_url']}")
        
        print("\nAgent Plan Details:")
        for idx, task in enumerate(data['plan']):
            print(f"  {idx+1}. Task ID: {task['id']}")
            print(f"     Description: {task['description']}")
            print(f"     Tool: {task['assigned_tool']}")
            print(f"     Status: {task['status']}")
            if task.get('section_heading'):
                print(f"     Section Heading: {task['section_heading']}")
            print(f"     Result Preview: {task['result'][:150]}...")
            
        print("\nAgent Log Trace:")
        for log in data['logs']:
            print(f"  [LOG] {log}")
            
        # Verify download
        download_url = f"{API_URL}{data['download_url']}"
        dl_response = httpx.get(download_url)
        assert dl_response.status_code == 200, f"Failed to download document from {download_url}"
        print(f"\nSUCCESS: Document successfully downloaded ({len(dl_response.content)} bytes)")
        print("=" * 60 + "\n")
        
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
