import argparse
import string
# import gymnasium as gym
import numpy as np
from jax import Array
import flax.linen as nn
import jax.numpy as jnp
import jax.random as random
import optax
from flax.core import FrozenDict
from flax.struct import dataclass
from jax import jit
from typing import Callable
from flax.training.train_state import TrainState
from flax import struct
from numpy import ndarray
import tensorflow_probability.substrates.jax.distributions as tfp
from tensorboardX import SummaryWriter
from jax import device_get
import time
# from tqdm import tqdm
# from termcolor import colored
import signal
from flax.training import orbax_utils
# import orbax.checkpoint
from jax.lax import stop_gradient
from jax import value_and_grad
from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv

class Actor(nn.Module):
    action_shape_prod: int

    @nn.compact
    def __call__(self, x: Array):
        action_mean = nn.Sequential([
            linear_layer_init(64),
            nn.tanh,
            linear_layer_init(64),
            nn.tanh,
            # linear_layer_init(64),
            # nn.tanh,
            linear_layer_init(self.action_shape_prod, std=0.01),
        ])(x)
        actor_logstd = self.param('logstd', nn.initializers.zeros, (1, self.action_shape_prod))
        action_logstd = jnp.broadcast_to(actor_logstd.squeeze(), action_mean.shape)  # Make logstd the same shape as actions
        return action_mean, action_logstd


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x: Array):
        return nn.Sequential([
            linear_layer_init(64),
            nn.tanh,
            linear_layer_init(64),
            nn.tanh,
            # linear_layer_init(64),
            # nn.tanh,
            linear_layer_init(1, std=1.0),
        ])(x)

@dataclass
class AgentParams:
    actor_params: FrozenDict
    critic_params: FrozenDict
    def __iter__(self):
        yield self.actor_params
        yield self.critic_params

class AgentState(TrainState):
    actor_fn: Callable = struct.field(pytree_node=False)
    critic_fn: Callable = struct.field(pytree_node=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-timesteps', type=int, default=150000, help='total timesteps of the experiment')
    parser.add_argument('--learning-rate', type=float, default=5e-3, help='the learning rate of the optimizer')
    parser.add_argument('--num-envs', type=int, default=1, help='the number of parallel environments')
    parser.add_argument('--num-steps', type=int, default=512,
                        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--gamma', type=float, default=0.96836, help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.99944,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=32, help='the number of mini batches')
    parser.add_argument('--update-epochs', type=int, default=10, help='the K epochs to update the policy')
    parser.add_argument('--clip-coef', type=float, default=0.2, help='the surrogate clipping coefficient')
    parser.add_argument('--ent-coef', type=float, default=0.0, help='coefficient of the entropy')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='coefficient of the value function')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='the maximum norm for the gradient clipping')
    parser.add_argument('--seed', type=int, default=1, help='seed for reproducible benchmarks')
    parser.add_argument('--exp-name', type=str, default='PPO_continuous_action', help='unique experiment name')
    parser.add_argument('--env-id', type=str, default='Reacher-v4', help='id of the environment')
    parser.add_argument('--capture-video', type=bool, default=False, help='whether to save video of agent gameplay')
    parser.add_argument('--track', type=bool, default=False, help='whether to track project with W&B')
    parser.add_argument("--wandb-project-name", type=str, default="RL-Flax", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")

    args = parser.parse_args()
    args.batch_size = args.num_envs * args.num_steps  # size of the batch after one rollout
    args.minibatch_size = args.batch_size // args.num_minibatches  # size of the mini batch
    args.num_updates = args.total_timesteps // args.batch_size  # the number of learning cycle

    return args


def make_env(env_id: string):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def linear_layer_init(features, std=np.sqrt(2), bias_const=0.0):
    layer = nn.Dense(features=features, kernel_init=nn.initializers.orthogonal(std),
                     bias_init=nn.initializers.constant(bias_const))
    return layer


def linear_schedule(count):
    frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
    return args.learning_rate * frac


@dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


# @jit
def get_action_and_value(agent_state: AgentState, next_obs: ndarray, next_done: ndarray, storage: Storage, step: int,
                         key: random.PRNGKey):
    action_mean, action_logstd = agent_state.actor_fn(agent_state.params.actor_params, next_obs)
    value = agent_state.critic_fn(agent_state.params.critic_params, next_obs)
    action_std = jnp.exp(action_logstd)
    probs = tfp.Normal(action_mean, action_std)
    key, subkey = random.split(key)
    action = probs.sample(seed=subkey)
    logprob = probs.log_prob(action).sum()
    storage = storage.replace(
        obs=storage.obs.at[step].set(next_obs),
        dones=storage.dones.at[step].set(next_done),
        actions=storage.actions.at[step].set(action),
        logprobs=storage.logprobs.at[step].set(logprob),
        values=storage.values.at[step].set(value.squeeze()),
    )
    return storage, action, key


@jit
def get_action_and_value2(agent_state: AgentState, params: AgentParams, obs: ndarray, action: ndarray):
    action_mean, action_logstd = agent_state.actor_fn(params.actor_params, obs)
    value = agent_state.critic_fn(params.critic_params, obs)
    action_std = jnp.exp(action_logstd)

    probs = tfp.Normal(action_mean, action_std)
    return probs.log_prob(action).sum(1), probs.entropy().sum(1), value.squeeze()


def rollout(
        agent_state: AgentState,
        next_obs: ndarray,
        next_done: ndarray,
        storage: Storage,
        key: random.PRNGKey,
        global_step: int,
        writer: SummaryWriter,
):
    prev_step = 0
    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        storage, action, key = get_action_and_value(agent_state, next_obs, next_done, storage, step, key)
        step_obs = envs.step(action)
        next_obs = step_obs.observation
        reward = step_obs.reward
        done = step_obs.done
        info = step_obs.info
        next_done = done
        storage = storage.replace(rewards=storage.rewards.at[step].set(reward))
        
        if(next_done):
            ep_return = storage.rewards.squeeze()[prev_step:step].sum()
            print(ep_return)
            writer.add_scalar("charts/episodic_return", 
                              ep_return, 
                              global_step)
            writer.add_scalar("charts/episodic_length", step - prev_step, global_step)
            print(f"Episode complete: global step {global_step}")
            prev_step = step
            next_obs = envs.reset()


    return next_obs, next_done, storage, key, global_step


# @jit
def compute_gae(
        agent_state: AgentState,
        next_obs: ndarray,
        next_done: ndarray,
        storage: Storage
):
    # Reset advantages values
    storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
    next_value = agent_state.critic_fn(agent_state.params.critic_params, next_obs).squeeze()
    # Compute advantage using generalized advantage estimate
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - storage.dones[t + 1]
            nextvalues = storage.values[t + 1]
        delta = storage.rewards[t] + args.gamma * nextvalues * nextnonterminal - storage.values[t]
        lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))

    storage = storage.replace(returns=storage.advantages + storage.values)
    return storage


@jit
def ppo_loss(
        agent_state: AgentState,
        params: AgentParams,
        obs: ndarray,
        act: ndarray,
        logp: ndarray,
        adv: ndarray,
        ret: ndarray,
        val: ndarray,
):
    newlogprob, entropy, newvalue = get_action_and_value2(agent_state, params, obs, act)
    logratio = newlogprob - logp
    ratio = jnp.exp(logratio)

    approx_kl = ((ratio - 1) - logratio).mean()

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    pg_loss1 = -adv * ratio
    pg_loss2 = -adv * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    v_loss_unclipped = (newvalue - ret) ** 2
    v_clipped = val + jnp.clip(
        newvalue - val,
        -args.clip_coef,
        args.clip_coef,
    )
    v_loss_clipped = (v_clipped - ret) ** 2
    v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
    v_loss = 0.5 * v_loss_max.mean()

    entropy_loss = entropy.mean()

    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    return loss, (pg_loss, v_loss, entropy_loss, stop_gradient(approx_kl))


def update_ppo(
        agent_state: AgentState,
        storage: Storage,
        key: random.PRNGKey
):
    # Flatten collected experiences
    b_obs = storage.obs.reshape((-1,) + envs.observation_space.shape)
    b_logprobs = storage.logprobs.reshape(-1)
    b_actions = storage.actions.reshape((-1,) + envs.action_space.shape)
    b_advantages = storage.advantages.reshape(-1)
    b_returns = storage.returns.reshape(-1)
    b_values = storage.values.reshape(-1)

    # Create function that will return gradient of the specified function
    ppo_loss_grad_fn = jit(value_and_grad(ppo_loss, argnums=1, has_aux=True))

    for _ in range(args.update_epochs):
        key, subkey = random.split(key)
        b_inds = random.permutation(subkey, args.batch_size, independent=True)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]
            (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                agent_state,
                agent_state.params,
                b_obs[mb_inds],
                b_actions[mb_inds],
                b_logprobs[mb_inds],
                b_advantages[mb_inds],
                b_returns[mb_inds],
                b_values[mb_inds],
            )

            agent_state = agent_state.apply_gradients(grads=grads)

    y_pred, y_true = b_values, b_returns
    var_y = jnp.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, explained_var, key


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.default_int_handler)
    args = parse_args()
    rand_state = np.random.RandomState(args.seed).get_state()
    np.random.set_state(rand_state)
    envs = ReacherEnv(
            setup="UR5_default",
            host="129.128.159.210",
            dof=2,
            control_type="velocity",
            target_type="position",
            reset_type="zero",
            reward_type="precision",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1.4,
            speed_max=0.3,
            speedj_a=1.4,
            episode_length_time=4.0,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=0.04,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0,
            random_state=rand_state
        )
    envs = NormalizedEnv(envs)
    envs.start()


    # envs = (make_env("InvertedPendulum-v4"))()
    # assert isinstance(envs.action_space, gym.spaces.Box), "only continuous action space is supported"
    obs = envs.reset()

    # Setting seed 
    key = random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    key, actor_key, critic_key, action_key, permutation_key = random.split(key, num=5)

    actor = Actor(action_shape_prod=np.array(envs.action_space.shape).prod()) # For jit we need to declare prod outside of class
    critic = Critic()

    # Probably jitting isn't needed as this functions should be jitted already
    actor.apply = jit(actor.apply)
    critic.apply = jit(critic.apply)

    # Initializing agent parameters
    actor_params = actor.init(actor_key, obs)
    critic_params = critic.init(critic_key, obs)

    tx = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=linear_schedule,
            eps=1e-5
        )
    )

    agent_state = AgentState.create(
        params=AgentParams(
            actor_params=actor_params,
            critic_params=critic_params
        ),
        tx=tx,
        # As we have separated actor and critic we don't use apply_fn
        apply_fn=None,
        actor_fn=actor.apply,
        critic_fn=critic.apply
    )

    run_name = f"{args.exp_name}_{args.seed}_{time.asctime(time.localtime(time.time())).replace('  ', ' ').replace(' ', '_')}"

    writer = SummaryWriter(f'runs/{args.env_id}/{run_name}')

    # Initialize the storage
    storage = Storage(
        obs=jnp.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape),
        actions=jnp.zeros((args.num_steps, args.num_envs) + envs.action_space.shape),
        logprobs=jnp.zeros((args.num_steps, args.num_envs)),
        dones=jnp.zeros((args.num_steps, args.num_envs)),
        values=jnp.zeros((args.num_steps, args.num_envs)),
        advantages=jnp.zeros((args.num_steps, args.num_envs)),
        returns=jnp.zeros((args.num_steps, args.num_envs)),
        rewards=jnp.zeros((args.num_steps, args.num_envs)),
    )
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = jnp.zeros(args.num_envs)
    
    try:
        for update in range(1, args.num_updates + 1):

            next_obs, next_done, storage, action_key, global_step = rollout(agent_state, next_obs, next_done, storage,
                                                                            action_key, global_step, writer)

            storage = compute_gae(agent_state, next_obs, next_done, storage)
            agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, explained_var, permutation_key = update_ppo(
                agent_state, storage, permutation_key)

            # writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(),
            #               global_step)
            # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            # writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print('Training complete!')
    except KeyboardInterrupt:
        print('Training canceled!')
    finally:
        envs.close()
        writer.close()
