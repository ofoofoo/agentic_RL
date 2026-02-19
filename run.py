import os
import argparse
from agent.agent import Agent
import yaml
from dotenv import load_dotenv

def main():
    load_dotenv()
    path = "config.yaml"
    with open(path) as f:
        config = yaml.safe_load(f)
    config["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY")
    agent = Agent(config)
    task = config["TASK"]
    agent.run(task)

if __name__ == "__main__":
    main()
