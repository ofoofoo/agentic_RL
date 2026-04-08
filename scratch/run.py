import os
import argparse
from agent.agent import Agent
import yaml
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Open Gmail")
    parser.add_argument(
        "--backend", type=str, default="gemini", choices=["gemini", "vllm"],
        help="Which model backend to use: 'gemini' (default) or 'vllm'",
    )
    args = parser.parse_args()
    load_dotenv()
    path = "config.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)
    config["BACKEND"] = args.backend
    config["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY")
    config["VLLM_API_KEY"] = os.environ.get("VLLM_API_KEY")
    agent = Agent(config)
    task = args.task
    agent.run(task)

if __name__ == "__main__":
    main()
