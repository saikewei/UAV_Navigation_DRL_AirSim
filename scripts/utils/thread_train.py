from .custom_policy_sb3 import CNN_FC, CNN_GAP, CNN_GAP_BN, No_CNN, CNN_MobileNet, CNN_GAP_new
import datetime
import gym
import gym_env
import numpy as np
import time
from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
import wandb
from PyQt5 import QtCore
import argparse
import ast
from configparser import ConfigParser
import torch as th
import os
import sys
from typing import List
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))


def get_parser():
    parser = argparse.ArgumentParser(
        description="Training thread without plot")
    parser.add_argument(
        '-c',
        '--config',
        help='config file name in configs folder, such as config_default',
        default='config_Trees_SimpleMultirotor_2D')
    parser.add_argument('-n',
                        '--note',
                        help='training objective',
                        default='depth_upper_split_5')

    return parser


class CheckpointReplayBufferCallback(BaseCallback):
    """Save model checkpoints and replay buffers at fixed step intervals."""

    def __init__(self, save_freq, save_path, save_replay_buffer=True,
                 total_timesteps=None, progress_log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.save_freq = max(1, int(save_freq))
        self.save_path = save_path
        self.save_replay_buffer = save_replay_buffer
        self.total_timesteps = total_timesteps
        self.progress_log_freq = max(1, int(progress_log_freq))

        self._train_start_time = None
        self._train_start_step = 0
        self._last_log_time = None
        self._last_log_step = 0

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _on_training_start(self) -> None:
        now = time.time()
        self._train_start_time = now
        self._last_log_time = now
        self._train_start_step = self.num_timesteps
        self._last_log_step = self.num_timesteps

    def _print_progress(self):
        if self._last_log_time is None or self._train_start_time is None:
            return

        now = time.time()
        current_step = self.num_timesteps
        step_delta = current_step - self._last_log_step
        time_delta = max(now - self._last_log_time, 1e-6)

        speed = step_delta / time_delta if step_delta > 0 else 0.0
        elapsed = max(now - self._train_start_time, 1e-6)
        elapsed_str = self._format_duration(elapsed)

        if self.total_timesteps is not None:
            remaining_steps = max(self.total_timesteps - current_step, 0)
            eta_seconds = remaining_steps / max(speed, 1e-6) if speed > 0 else 0
            eta_str = self._format_duration(eta_seconds)
            print(
                "train progress | step: {} / {} | speed: {:.2f} steps/s | elapsed: {} | eta: {}".format(
                    current_step, self.total_timesteps, speed, elapsed_str, eta_str
                )
            )
        else:
            print(
                "train progress | step: {} | speed: {:.2f} steps/s | elapsed: {}".format(
                    current_step, speed, elapsed_str
                )
            )

        self._last_log_time = now
        self._last_log_step = current_step

    def _on_step(self) -> bool:
        if self.model is None:
            return True

        if self.n_calls % self.progress_log_freq == 0:
            self._print_progress()

        if self.n_calls % self.save_freq != 0:
            return True

        step = self.num_timesteps
        checkpoint_path = os.path.join(
            self.save_path, f'checkpoint_{step}_steps')
        latest_checkpoint_path = os.path.join(self.save_path, 'checkpoint_latest')

        self.model.save(checkpoint_path)
        self.model.save(latest_checkpoint_path)

        save_replay_buffer_fn = getattr(self.model, 'save_replay_buffer', None)
        if self.save_replay_buffer and callable(save_replay_buffer_fn):
            replay_buffer_path = os.path.join(
                self.save_path, f'replay_buffer_{step}_steps.pkl')
            latest_replay_buffer_path = os.path.join(
                self.save_path, 'replay_buffer_latest.pkl')

            save_replay_buffer_fn(replay_buffer_path)
            save_replay_buffer_fn(latest_replay_buffer_path)

        if self.verbose > 0:
            print(f'checkpoint saved at step {step}')

        return True


class TrainingThread(QtCore.QThread):
    """
    QT thread for policy training
    """

    def __init__(self, config):
        super(TrainingThread, self).__init__()
        print("init training thread")

        # config
        self.cfg = ConfigParser()
        self.cfg.read(config)

        env_name = self.cfg.get('options', 'env_name')
        self.project_name = env_name

        # make gym environment
        self.env = gym.make('airsim-env-v0')
        self.env.set_config(self.cfg)

        wandb_name = self.cfg.get(
            'options', 'policy_name') + '-' + self.cfg.get('options', 'algo')
        if self.cfg.get('options', 'dynamic_name') == 'SimpleFixedwing':
            if self.cfg.get('options', 'perception') == "lgmd":
                wandb_name += '-LGMD'
            else:
                wandb_name += '-depth'
            if self.cfg.getfloat('fixedwing', 'pitch_flap_hz') != 0:
                wandb_name += '-Flapping'

        # wandb
        if self.cfg.getboolean('options', 'use_wandb'):
            wandb.init(
                project=self.project_name,
                notes=self.cfg.get('wandb', 'notes'),
                name=self.cfg.get('wandb', 'name'),
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                save_code=True,  # optional
            )

    def terminate(self):
        print('TrainingThread terminated')

    def _cfg_getboolean(self, section, option, default):
        if self.cfg.has_option(section, option):
            return self.cfg.getboolean(section, option)
        return default

    def _cfg_getint(self, section, option, default):
        if self.cfg.has_option(section, option):
            return self.cfg.getint(section, option)
        return default

    def _cfg_get(self, section, option, default):
        if self.cfg.has_option(section, option):
            return self.cfg.get(section, option)
        return default

    def _resolve_resume_run_path(self, resume_model_path):
        """Resolve experiment root directory from a checkpoint/model path."""
        path = os.path.abspath(resume_model_path)

        if os.path.isdir(path):
            if os.path.basename(path) == 'models':
                return os.path.dirname(path)
            if os.path.isdir(os.path.join(path, 'tb_logs')):
                return path
            return None

        model_dir = os.path.dirname(path)
        if os.path.basename(model_dir) == 'models':
            return os.path.dirname(model_dir)

        return None

    def run(self):
        print("run training thread")

        resume_training = self._cfg_getboolean(
            'checkpoint', 'resume_training', False)
        resume_model_path = self._cfg_get(
            'checkpoint', 'model_path', '').strip()
        resume_replay_buffer_path = self._cfg_get(
            'checkpoint', 'replay_buffer_path', '').strip()

        # ! -----------------------------------init folders-----------------------------------------
        if resume_training:
            if resume_model_path == '':
                raise Exception(
                    'resume_training=True but checkpoint.model_path is empty')

            resume_run_path = self._resolve_resume_run_path(resume_model_path)
            if resume_run_path is None:
                raise Exception(
                    'cannot resolve run folder from checkpoint.model_path, expected .../models/<model_file>.zip')

            file_path = resume_run_path
            print('reuse run folder for resume:', file_path)
        else:
            now = datetime.datetime.now()
            now_string = now.strftime('%Y_%m_%d_%H_%M')
            file_path = 'logs/' + self.project_name + '/' + now_string + '_' + self.cfg.get(
                'options', 'dynamic_name') + '_' + self.cfg.get(
                    'options', 'policy_name') + '_' + self.cfg.get(
                        'options', 'algo')

        log_path = file_path + '/tb_logs'
        model_path = file_path + '/models'
        config_path = file_path + '/config'
        data_path = file_path + '/data'
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(config_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)  # create data path to save q_map

        # save config file
        with open(os.path.join(config_path, 'config.ini'), 'w') as configfile:
            self.cfg.write(configfile)

        #! -----------------------------------policy selection-------------------------------------
        feature_num_state = self.env.dynamic_model.state_feature_length
        feature_num_cnn = self.cfg.getint('options', 'cnn_feature_num')
        policy_name = self.cfg.get('options', 'policy_name')

        # feature extraction network
        if self.cfg.get('options', 'activation_function') == 'tanh':
            activation_function = th.nn.Tanh
        else:
            activation_function = th.nn.ReLU
        
        if policy_name == 'mlp':
            policy_base = 'MlpPolicy'
            policy_kwargs = dict(activation_fn=activation_function)
        else:
            policy_base = 'CnnPolicy'
            if policy_name == 'CNN_FC':
                policy_used = CNN_FC
            elif policy_name == 'CNN_GAP':
                policy_used = CNN_GAP_new
            elif policy_name == 'CNN_GAP_BN':
                policy_used = CNN_GAP_BN
            elif policy_name == 'CNN_MobileNet':
                policy_used = CNN_MobileNet
            elif policy_name == 'No_CNN':
                policy_used = No_CNN
            else:
                raise Exception('policy select error: ', policy_name)

            policy_kwargs = dict(
                features_extractor_class=policy_used,
                features_extractor_kwargs=dict(
                    features_dim=feature_num_state + feature_num_cnn,
                    state_feature_dim=feature_num_state),
                activation_fn=activation_function)

        # fully-connected work after feature extraction
        net_arch_list = ast.literal_eval(self.cfg.get("options", "net_arch"))
        policy_kwargs['net_arch'] = net_arch_list

        #! ---------------------------------algorithm selection-------------------------------------
        algo = self.cfg.get('options', 'algo')
        print('algo: ', algo)
        if algo == 'PPO':
            model = PPO(
                policy_base,
                self.env,
                # n_steps = 200,
                learning_rate=self.cfg.getfloat('DRL', 'learning_rate'),
                policy_kwargs=policy_kwargs,
                tensorboard_log=log_path,
                seed=0,
                verbose=2)
        elif algo == 'SAC':
            n_actions = self.env.action_space.shape[-1]
            noise_sigma = self.cfg.getfloat(
                'DRL', 'action_noise_sigma') * np.ones(n_actions)
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=noise_sigma)
            model = SAC(
                policy_base,
                self.env,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                buffer_size=self.cfg.getint('DRL', 'buffer_size'),
                gamma=self.cfg.getfloat('DRL', 'gamma'),
                learning_starts=self.cfg.getint('DRL', 'learning_starts'),
                learning_rate=self.cfg.getfloat('DRL', 'learning_rate'),
                batch_size=self.cfg.getint('DRL', 'batch_size'),
                train_freq=(self.cfg.getint('DRL', 'train_freq'), 'step'),
                gradient_steps=self.cfg.getint('DRL', 'gradient_steps'),
                tensorboard_log=log_path,
                seed=0,
                verbose=2)
        elif algo == 'TD3':
            # The noise objects for TD3
            n_actions = self.env.action_space.shape[-1]
            noise_sigma = self.cfg.getfloat(
                'DRL', 'action_noise_sigma') * np.ones(n_actions)
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=noise_sigma)
            model = TD3(
                policy_base,
                self.env,
                action_noise=action_noise,
                learning_rate=self.cfg.getfloat('DRL', 'learning_rate'),
                gamma=self.cfg.getfloat('DRL', 'gamma'),
                policy_kwargs=policy_kwargs,
                learning_starts=self.cfg.getint('DRL', 'learning_starts'),
                batch_size=self.cfg.getint('DRL', 'batch_size'),
                train_freq=(self.cfg.getint('DRL', 'train_freq'), 'step'),
                gradient_steps=self.cfg.getint('DRL', 'gradient_steps'),
                buffer_size=self.cfg.getint('DRL', 'buffer_size'),
                tensorboard_log=log_path,
                seed=0,
                verbose=2)
        else:
            raise Exception('Invalid algo name : ', algo)

        #! ---------------------------------resume training-------------------------------------
        if resume_training:
            print('resume from checkpoint:', resume_model_path)
            if algo == 'PPO':
                model = PPO.load(
                    resume_model_path,
                    env=self.env,
                    tensorboard_log=log_path,
                    seed=0,
                    verbose=2)
            elif algo == 'SAC':
                model = SAC.load(
                    resume_model_path,
                    env=self.env,
                    tensorboard_log=log_path,
                    seed=0,
                    verbose=2)
            elif algo == 'TD3':
                model = TD3.load(
                    resume_model_path,
                    env=self.env,
                    tensorboard_log=log_path,
                    seed=0,
                    verbose=2)

            if resume_replay_buffer_path != '':
                load_replay_buffer_fn = getattr(model, 'load_replay_buffer', None)
                if callable(load_replay_buffer_fn):
                    load_replay_buffer_fn(resume_replay_buffer_path)
                    print('replay buffer loaded:', resume_replay_buffer_path)
                else:
                    print('current algo has no replay buffer, skip loading replay buffer')

        # TODO create eval_callback
        # eval_freq = self.cfg.getint('TD3', 'eval_freq')
        # n_eval_episodes = self.cfg.getint('TD3', 'n_eval_episodes')
        # eval_callback = EvalCallback(self.env, best_model_save_path= file_path + '/eval',
        #                      log_path= file_path + '/eval', eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
        #                      deterministic=True, render=False)

        #! -------------------------------------train-----------------------------------------
        print('start training model')
        total_timesteps = self.cfg.getint('options', 'total_timesteps')
        self.env.model = model
        self.env.data_path = data_path

        checkpoint_save_freq = self._cfg_getint('checkpoint', 'save_freq', 5000)
        progress_log_freq = self._cfg_getint('checkpoint', 'progress_log_freq', 1000)
        save_replay_buffer = self._cfg_getboolean(
            'checkpoint', 'save_replay_buffer', True)

        current_model_steps = model.num_timesteps
        if resume_training:
            total_target_steps = current_model_steps + total_timesteps
        else:
            total_target_steps = total_timesteps

        callbacks: List[BaseCallback] = []
        callbacks.append(
            CheckpointReplayBufferCallback(
                save_freq=checkpoint_save_freq,
                save_path=model_path,
                save_replay_buffer=save_replay_buffer,
                total_timesteps=total_target_steps,
                progress_log_freq=progress_log_freq,
                verbose=1,
            )
        )

        if self.cfg.getboolean('options', 'use_wandb'):
            callbacks.append(
                WandbCallback(
                    model_save_freq=0,
                    gradient_save_freq=5000,
                    model_save_path=model_path,
                    verbose=2,
                )
            )

        callback_used = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)
        reset_num_timesteps = not resume_training
        tb_log_name = self._cfg_get(
            'checkpoint', 'tb_log_name',
            self.cfg.get('options', 'dynamic_name') + '_' + self.cfg.get('options', 'policy_name') + '_' + algo
        )

        if self.cfg.getboolean('options', 'use_wandb'):
            # if algo == 'TD3' or algo == 'SAC':
            #     wandb.watch(model.actor, log_freq=100, log="all")  # log gradients
            # elif algo == 'PPO':
            #     wandb.watch(model.policy, log_freq=100, log="all")
            model.learn(
                total_timesteps,
                log_interval=1,
                callback=callback_used,
                tb_log_name=tb_log_name,
                reset_num_timesteps=reset_num_timesteps,
            )
        else:
            model.learn(
                total_timesteps,
                log_interval=1,
                callback=callback_used,
                tb_log_name=tb_log_name,
                reset_num_timesteps=reset_num_timesteps,
            )

        #! ---------------------------model save----------------------------------------------------
        model_name = 'model_sb3'
        model.save(model_path + '/' + model_name)

        save_replay_buffer_fn = getattr(model, 'save_replay_buffer', None)
        if save_replay_buffer and callable(save_replay_buffer_fn):
            save_replay_buffer_fn(model_path + '/replay_buffer_final.pkl')

        print('training finished')
        print('model saved to: {}'.format(model_path))


def main():
    parser = get_parser()
    args = parser.parse_args()

    config_file = 'configs/' + args.config + '.ini'

    print(config_file)

    training_thread = TrainingThread(config_file)
    training_thread.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
