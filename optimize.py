
import numpy as np
from sac import SAC
import os, pickle
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import logging, yaml
import gym
log = logging.getLogger(__name__)
# import pybullet as p
# import pybullet_data
# from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv, PusherBulletEnv
# from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
# from gym.envs.mujoco import mujoco_env
# import mujoco_py
import hydra
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
sys.path.insert(0, parent_dir+"/VREnv/") 
gym.envs.register(
     id='VREnv-v0',
     entry_point='VREnv.src.envs.play_table_env:PlayTableSimEnv',
     max_episode_steps=200,
)

class SACWorker(Worker):
    def __init__(self, hyperparameters, eval_config, learn_config, run_id, nameserver):
        super(SACWorker,self).__init__(run_id, nameserver = nameserver)
        self.hyperparameters = hyperparameters
        self.eval_config = eval_config
        self.learn_config = learn_config
    
    def compute(self, config, budget, working_directory, *args, **kwargs):
        log.info("Running job/config_id: " + str(kwargs["config_id"]) )
        log.info(config)
        model = SAC(**config, **self.hyperparameters)
        stats = model.learn(total_timesteps = int(budget), **self.learn_config)
        train_reward = np.max(stats.episode_rewards)
        eval_mean_reward, _ = model.evaluate(model.eval_env, **self.eval_config)

        return ({ #want to maximize reward then minimize negative reward
                'loss': - eval_mean_reward, # remember: HpBandSter always minimizes!
                'info': { 'Max train reward': train_reward,
                          'Mean validation reward': eval_mean_reward,
                        } })
    
    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()
        actor_lr = CSH.UniformFloatHyperparameter('actor_lr', lower=1e-6, upper=1e-2, log=True)
        critic_lr = CSH.UniformFloatHyperparameter('critic_lr', lower=1e-6, upper=1e-2, log=True)
        alpha_lr = CSH.UniformFloatHyperparameter('alpha_lr', lower=1e-6, upper=1e-2, log=True)
        #alpha = CSH.UniformFloatHyperparameter('alpha', lower=0.01, upper=0.7)
        tau = CSH.UniformFloatHyperparameter('tau', lower=0.001, upper=0.02)
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=128, upper=256)
        hidden_dim = CSH.UniformIntegerHyperparameter('hidden_dim', lower=256, upper=512)

        cs.add_hyperparameters([actor_lr, critic_lr, alpha_lr, tau, batch_size, hidden_dim])
        return cs

def optimize(trial_name, hyperparameters, eval_config, learn_config,\
                max_budget = 250000, min_budget = 50000, n_iterations = 3, n_workers=1):  

    NS = hpns.NameServer(run_id='sac_hpo', host='127.0.0.1', port=None)
    NS.start()
    workers=[]
    #for i in range(n_workers):
    w = SACWorker(hyperparameters, eval_config, learn_config,\
                    nameserver='127.0.0.1', run_id='sac_hpo')#, id=i)
    w.run(background=True)
    #workers.append(w)

    bohb = BOHB( configspace = w.get_configspace(),
            run_id = 'sac_hpo', nameserver='127.0.0.1',
            min_budget=min_budget, max_budget=max_budget )
    res = bohb.run(n_iterations=n_iterations) #, min_workers = n_workers)
    # store results
    if not os.path.exists("./optimization_results/"): 
            os.makedirs("./optimization_results/")
    with open(os.path.join("./optimization_results/", "%s.pkl"%trial_name), 'wb') as fh:
        pickle.dump(res, fh)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()
    id2config = res.get_id2config_mapping()
    incumbent = str(res.get_incumbent_id())
    print("incumbent: ", incumbent)
    log.info("Finished optimization, incumbent: %s"%incumbent)

def read_results(name):
    with open(os.path.join("./optimization_results/", name), 'rb') as fh:
        res = pickle.load(fh)

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    print(id2config[incumbent])

def load_env_config(config_path = "./config/config.yaml"):
    config  = yaml.load(open(config_path, 'r'))
    return config["env"]

def load_agent_config(config_path = "./config/config.yaml"):
    config  = yaml.load(open(config_path, 'r'))
    agent_config =  config["agent"]["hyperparameters"]
    agent_config["save_dir"] =  config["agent"]["save_dir"]
    learn_config = config["agent"]["learn_configuration"]
    return agent_config, learn_config

@hydra.main(config_path="./config", config_name="config_vrenv")
def optim_vrenv(cfg):
    model_name = cfg.model_name
    hyperparameters = cfg.optim.hyperparameters
    learn_config = cfg.optim.learn_config
    eval_config =  cfg.optim.eval_config
    hp = {}
    hp["env"] = gym.make("VREnv-v0", **cfg.env).env
    hp["eval_env"] = gym.make("VREnv-v0", **cfg.eval_env).env
    hp["model_name"] = model_name
    hp = {**hyperparameters, **hp}
    optimize(model_name, hp, eval_config, learn_config,\
            max_budget=cfg.optim.max_budget, min_budget=cfg.optim.min_budget,\
            n_iterations = cfg.optim.n_iterations, n_workers=cfg.optim.n_workers)
    read_results("%s.pkl"%model_name)

def optim_gymenv(env_name, model_name):
    hyperparameters = {
        "gamma": 0.98,
        "buffer_size": 1e6,
        "train_freq": 1, #timesteps collectng data
        "gradient_steps": 1, #timesteps updating gradients
        "learning_starts": 1000, #timesteps before starting updates
    }
    eval_config = {
        "max_episode_length": 1000, 
        "n_episodes": 50,
        "render": False,
        "print_all_episodes": False,
        "write_file": False,
    }
    learn_configuration = {
        "log_interval": 1000, #log timestep reward every log_interval steps
        "max_episode_length": 200, #max episode length
    }
    hyperparameters["env"] = gym.make(env_name).env
    hyperparameters["eval_env"] = gym.make(env_name).env
    hyperparameters["model_name"] = model_name
    
    optimize(model_name, hyperparameters, eval_config,\
             learn_configuration, n_iterations = 2)
    read_results("%s.pkl"%model_name)

if __name__ == "__main__":
    # env_name = "Pusher-v2"
    # model_name = "sac_vrenv200steps"
    #optim_gymenv(env_name, model_name)
    optim_vrenv()