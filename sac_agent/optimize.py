
import numpy as np
import os, pickle
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import logging
log = logging.getLogger(__name__)
import gym
import hydra
import os,sys

parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, os.getcwd())
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/VREnv/")
from sac_agent.sac import SAC
from utils.env_processing_wrapper import EnvWrapper
gym.envs.register(
     id='VREnv-v0',
     entry_point='VREnv.vr_env.envs.play_table_env:PlayTableSimEnv',
     max_episode_steps=200,
)


class SACWorker(Worker):
    def __init__(self, hyperparameters, eval_config, learn_config, run_id, nameserver, logger):
        super().__init__(run_id, nameserver, logger = logger)
        self.hyperparameters = hyperparameters
        self.eval_config = eval_config
        self.learn_config = learn_config
    
    def compute(self, config, budget, working_directory, *args, **kwargs):
        log.info("Running job/config_id: " + str(kwargs["config_id"]) )
        log.info(config)
        if("hidden_dim" in config):
            self.hyperparameters['net_cfg'].update({"hidden_dim":config.pop("hidden_dim")})
        model = SAC(**config, **self.hyperparameters)
        stats = model.learn(total_timesteps = int(budget), **self.learn_config)
        train_reward = np.max(stats.episode_rewards)
        eval_mean_reward, _ = model.evaluate(model.eval_env, **self.eval_config)

        if(len(stats.validation_reward)>0):
            max_validation_reward = np.max(stats.validation_reward).item()
        else:
            max_validation_reward = eval_mean_reward

        return ({ #want to maximize reward then minimize negative reward
                'loss': - eval_mean_reward, # remember: HpBandSter always minimizes!
                'info': { 'Max train reward': train_reward,
                          'Max validation reward': max_validation_reward,
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
        actor_lr = CSH.UniformFloatHyperparameter('actor_lr', lower=1e-6, upper=1e-3, log=True)
        critic_lr = CSH.UniformFloatHyperparameter('critic_lr', lower=1e-6, upper=1e-3, log=True)
        alpha_lr = CSH.UniformFloatHyperparameter('alpha_lr', lower=1e-6, upper=1e-3, log=True)
        tau = CSH.UniformFloatHyperparameter('tau', lower=0.001, upper=0.01)
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=32, upper=128)
        hidden_dim = CSH.UniformIntegerHyperparameter('hidden_dim', lower=128, upper=512)

        cs.add_hyperparameters([actor_lr, critic_lr, alpha_lr, hidden_dim, tau, batch_size])
        return cs

def optimize(trial_name, hyperparameters, eval_config, learn_config, run_id, nameserver,\
                max_budget = 250000, min_budget = 50000, n_iterations = 3, n_workers=1, worker=False):  
    
    result_logger = hpres.json_result_logger(directory=".", overwrite=False)
    logging.basicConfig(level=logging.DEBUG)

    # Start a nameserver (see example_1)
    NS = hpns.NameServer(run_id=run_id, host=nameserver, port=None)
    NS.start()

    w = SACWorker(hyperparameters, eval_config, learn_config,\
                nameserver = nameserver, run_id = run_id, logger = log)#, id=i)
    w.run(background=True)

    bohb = BOHB( configspace = w.get_configspace(),
                 run_id = run_id,
                 nameserver = nameserver,
                 result_logger=result_logger,
                 min_budget=min_budget, 
                 max_budget=max_budget )

    res = bohb.run(n_iterations=n_iterations)
                   # , min_n_workers = n_workers)
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

@hydra.main(config_path="../config", config_name="cfg_sac")
def optim_vrenv(cfg):
    model_name = cfg.model_name
    hyperparameters = cfg.optim.hyperparameters
    learn_config = cfg.optim.learn_config
    eval_config =  cfg.optim.eval_config
    hp = {}
    hp["env"] = gym.make("VREnv-v0", **cfg.env).env
    hp["eval_env"] = gym.make("VREnv-v0", **cfg.eval_env).env
    hp["model_name"] = model_name
    hp["img_obs"] = cfg.img_obs
    hp["env"] = EnvWrapper(hp["env"], **cfg.env_wrapper)
    hp["eval_env"] = EnvWrapper(hp["eval_env"], **cfg.env_wrapper)

    net_cfg = {}
    for k,v in cfg.optim.net_cfg.items():
        net_cfg[k] = v
    hp = {**hyperparameters, **hp, 'net_cfg': net_cfg }
    worker=False
    if(cfg.optim.n_workers>1):
        worker=True
    optimize(model_name, hp, eval_config, learn_config, 
                run_id = cfg.optim.run_id, 
                nameserver = cfg.optim.nameserver,\
                max_budget=cfg.optim.max_budget, 
                min_budget=cfg.optim.min_budget,\
                n_iterations = cfg.optim.n_iterations, 
                n_workers=cfg.optim.n_workers,
                worker=worker)
    read_results("%s.pkl"%model_name)


if __name__ == "__main__":
    optim_vrenv()