import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state import create_initial_state
from src.graph import build_agent_graph

def trace_graph():
    initial_state = create_initial_state()
    initial_state["_pdf_path"] = "patient 2 (1).pdf"
    
    agent = build_agent_graph()
    
    print("Starting LangGraph streaming trace...")
    step_count = 0
    for event in agent.stream(initial_state, {"recursion_limit": 150}):
        step_count += 1
        for node_name, node_output in event.items():
            print(f"Event {step_count}: Node '{node_name}' output keys: {list(node_output.keys())}")
            if "steps_remaining" in node_output:
                print(f"  -> steps_remaining in output: {node_output['steps_remaining']}")
            
            # Print trace entries added by this node
            if "trace" in node_output:
                for entry in node_output["trace"]:
                    print(f"  -> Trace entry: step={entry.get('step')}, phase={entry.get('phase')}, action={entry.get('action')}, steps_remaining={entry.get('reasoning')}")

if __name__ == "__main__":
    trace_graph()
