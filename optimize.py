import gym
# import pybullet as p
# import pybullet_data
# from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv, PusherBulletEnv
# from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from gym.envs.mujoco import mujoco_env
import mujoco_py
import numpy as np
from sac import SAC
from datetime import datetime
import time
from utils.networks import ActorNetwork, CriticNetwork
import os, pickle
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import logging
from sac import SAC

class SACWorker(Worker):
    def __init__(self, hyperparameters, eval_config, learn_config, run_id, nameserver):
        super(SACWorker,self).__init__(run_id, nameserver = nameserver)
        self.hyperparameters = hyperparameters
        self.eval_config = eval_config
        self.learn_config = learn_config
    
    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """
        model = SAC(**config, **self.hyperparameters)
        stats = model.learn(total_timesteps = int(budget), **learn_config)
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
        actor_lr = CSH.UniformFloatHyperparameter('actor_lr', lower=1e-6, upper=1e-3, log=True)
        critic_lr = CSH.UniformFloatHyperparameter('critic_lr', lower=1e-6, upper=1e-3, log=True)
        alpha_lr = CSH.UniformFloatHyperparameter('alpha_lr', lower=1e-6, upper=1e-3, log=True)
        #alpha = CSH.UniformFloatHyperparameter('alpha', lower=0.01, upper=0.7)
        tau = CSH.UniformFloatHyperparameter('tau', lower=0.001, upper=0.02)
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=128, upper=256)
        hidden_dim = CSH.UniformIntegerHyperparameter('hidden_dim', lower=256, upper=512)

        cs.add_hyperparameters([actor_lr, critic_lr, alpha_lr, tau, batch_size, hidden_dim])
        return cs

def optimize(trial_name, hyperparameters, eval_config, learn_config,\
                max_budget = 250000, min_budget = 50000, n_iterations = 4):  
    # Step 1: Start a nameserver
    # Every run needs a nameserver. It could be a 'static' server with a
    # permanent address, but here it will be started for the local machine with the default port.
    # The nameserver manages the concurrent running workers across all possible threads or clusternodes.
    # Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
    NS = hpns.NameServer(run_id='sac_hpo', host='127.0.0.1', port=None)
    NS.start()

    # Step 2: Start a worker
    # Now we can instantiate a worker, providing the mandatory information
    # Besides the sleep_interval, we need to define the nameserver information and
    # the same run_id as above. After that, we can start the worker in the background,
    # where it will wait for incoming configurations to evaluate.
    w = SACWorker(hyperparameters, eval_config, learn_config, nameserver='127.0.0.1', run_id='sac_hpo')
    w.run(background=True)

    # Step 3: Run an optimizer
    # Now we can create an optimizer object and start the run.
    # Here, we run BOHB, but that is not essential.
    # The run method will return the `Result` that contains all runs performed.
    bohb = BOHB(  configspace = w.get_configspace(),
            run_id = 'sac_hpo', nameserver='127.0.0.1',
            min_budget=min_budget, max_budget=max_budget
        )
    res = bohb.run(n_iterations=n_iterations)
    
    # store results
    with open(os.path.join("./optimization_results/", "%s.pkl"%trial_name), 'wb') as fh:
        pickle.dump(res, fh)
    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # Step 5: Analysis
    # Each optimizer returns a hpbandster.core.result.Result object.
    # It holds informations about the optimization run like the incumbent (=best) configuration.
    # For further details about the Result object, see its documentation.
    # Here we simply print out the best config and some statistics about the performed runs.
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

def read_results(name):
    with open(os.path.join("./optimization_results/", name), 'rb') as fh:
        res = pickle.load(fh)

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()
    print(id2config[incumbent])

if __name__ == "__main__":
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

    env_name = "Pusher-v2"
    hyperparameters["env"] = gym.make(env_name).env
    hyperparameters["eval_env"] = gym.make(env_name).env
    hyperparameters["model_name"] = "sac_mujocoPusher_200steps"
    
    optimize("sac_mujocoPusher_200steps", hyperparameters, eval_config, learn_configuration)
    read_results("sac_mujocoPusher_200steps.pkl")
