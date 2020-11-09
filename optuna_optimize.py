import numpy as np
import optuna
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import RandomSampler, TPESampler
from torch import nn as nn

def hyperparam_optimization(eval_env, hyperparams, eval_config, learn_config, \
                            n_startup_trials = 10, seed = 0, study_name = None):
    # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
    #sampler = RandomSampler(seed=seed)
    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)

    #pruner = MedianPruner(n_startup_trials=n_trials, n_warmup_steps=n_evaluations)
    #pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3)
    study = optuna.create_study(sampler=sampler, pruner=pruner, study_name=study_name, load_if_exists=True, direction="maximize" )
    algo_sampler = sample_sac_params

    def objective(trial):
        kwargs = hyperparams.copy()

        trial.model_class = None
        kwargs.update(algo_sampler(trial))

        model = SAC(**hyperparameters)
        model.trial = trial

        eval_env = env_fn()
        try:
            model.learn(n_timesteps)
            # Free memory
            model.env.close()
            eval_env.close()
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        mean_reward, _ = model.evaluate(eval_env, **eval_config)
        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.trials_dataframe()

def sample_sac_params(trial):
    """
    Sampler for SAC hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    actor_lr = trial.suggest_loguniform("actor_lr", 1e-5, 1)
    critic_lr = trial.suggest_loguniform("critic_lr", 1e-5, 1)
    alpha_lr = trial.suggest_loguniform("alpha_lr", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02])

    return {
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "alpha_lr": alpha_lr,
        "batch_size": batch_size,
        "tau": tau,
        "target_entropy": target_entropy,
    }