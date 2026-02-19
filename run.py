import argparse
from agent.agent import Agent
from agent.config import load_config

def main():
    parser = argparse.ArgumentParser(description="agentic_RL — Gemini + Android agent")
    parser.add_argument("--task", type=str, help="Task for the agent to complete")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()
    print(f"args config:{args.config}")

    config = load_config(args.config)
    agent = Agent(config)

    task = args.task or input("Enter task: ").strip()
    agent.run(task)

if __name__ == "__main__":
    main()
