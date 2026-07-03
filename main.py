import argparse
import os 
import time 
from utils.helper import loadConfig
from trainers.train_sac import SACTrainer
from trainers.train_ppo import PPOTrainer

def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="SAC")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()

def process_configDir(agent, configDir=None): 
    if configDir is None: 
        if agent.lower() not in ["sac", "sacher", "ppo"]: 
            raise ValueError(f"Supported `SAC`, `SACHER` and `PPO`, found {agent}")
        return f"configs/{agent.upper()}.yaml"
    else: 
        if not os.path.exists(configDir): 
            raise FileNotFoundError(f"Cannot find configuration file {configDir}")
        return configDir

def main(): 
    args = get_args()
    configDir = process_configDir(args.agent, args.config)
    config = loadConfig(configDir)
    start_time = time.perf_counter()
    if args.agent.lower()=="sac": 
        agent= SACTrainer(config)
    elif args.agent.lower()=="sacher": 
        # TODO : Imlement SAC HER agent
        pass 
    elif args.agent.lower()=="ppo":
        agent = PPOTrainer(config)
    else:
        raise ValueError(f"Unsupported agent: {args.agent}")
    agent.train()
    end_time = time.perf_counter()-start_time
    print(f"Total runtime : {end_time}s")

if __name__ == "__main__": 
    main()
