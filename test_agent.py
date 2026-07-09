import os
from dotenv import load_dotenv
from agent import run_agent

def main():
    load_dotenv()
    print("Starting integration test for Autonomous Document Agent...")
    
    # Simple test request that triggers the agent
    request = "Write a short proposal for a local community garden. Include a table showing estimated costs for soil, seeds, and tools."
    
    try:
        result = run_agent(request)
        
        print("\n--- Test Execution Completed ---")
        print("Document Title:", result.get("document_title"))
        print("Final Doc Path:", result.get("final_doc_path"))
        
        # Print final plan tasks
        print("\nFinal Plan Tasks:")
        for task in result.get("plan", []):
            print(f"- {task['id']} ({task['status']}) - Tool: {task['assigned_tool']}")
            
        print("\nLogs Trace:")
        for log in result.get("logs", []):
            print(" ", log)
            
        # Assertions
        assert result.get("document_title") != "", "Document title should not be empty"
        assert len(result.get("plan", [])) > 0, "Plan should contain tasks"
        assert result.get("final_doc_path") != "", "Final document path should not be empty"
        assert os.path.exists(result.get("final_doc_path")), f"File does not exist: {result.get('final_doc_path')}"
        
        print("\nSUCCESS: Integration test passed successfully! Generated file is valid.")
        
    except Exception as e:
        print("\nFAILURE: Integration test encountered an error:", e)
        raise e

if __name__ == "__main__":
    main()
