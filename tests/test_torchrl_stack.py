"""Card 02 gate: the pinned TorchRL stack handles a masked Dict-observation env end to end on CPU.

Uses a dummy env with the same interface as the real WasteFlow env (Dict obs with "observation" and an
int8 "action_mask", Discrete(17), action 16 = wait is always valid). Each test exercises one piece the
training code will rely on: env wrapping + spec checks, masked random rollout, Collector + PPO/A2C/REINFORCE
losses with MaskedCategorical, and DQN with masked Q-values.

Findings for torchrl 0.15.0 (recorded on the "Approved design decisions" card):
    - SyncDataCollector is gone; use torchrl.collectors.Collector.
    - GymEnv defaults to a OneHot action spec; pass categorical_action_encoding=True for integer actions.
    - The gym MultiBinary mask arrives as int8 and must be cast to bool before ActionMask / MaskedCategorical.
    - QValueActor / QValueModule / EGreedyModule accept action_mask_key natively (no custom module needed).
    - Give MLPs an explicit in_features: a lazy critic that the collector never calls is still uninitialized
      when the loss module is built, and the loss fails.
    - Collector warns about InitTracker; pass auto_register_policy_transforms=False (no recurrent policies).
"""

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import ActionMask, Compose, DTypeCastTransform
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MLP, EGreedyModule, MaskedCategorical, ProbabilisticActor, QValueActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, DQNLoss, HardUpdate, ReinforceLoss
from torchrl.objectives.value import GAE

N_ACTIONS = 17
OBS_DIM = 8
ENV_ID = "WasteFlowSpike-v0"


class SpikeEnv(gym.Env):
    """Interface stand-in for the WasteFlow env: random valid-action mask, +1 for valid actions, -1 otherwise."""

    def __init__(self, max_steps: int = 20):
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32),
                "action_mask": spaces.MultiBinary(N_ACTIONS),
            }
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.max_steps = max_steps

    def _obs(self):
        mask = self.np_random.integers(0, 2, N_ACTIONS).astype(np.int8)
        mask[N_ACTIONS - 1] = 1  # wait is always valid
        self._mask = mask
        return {"observation": self.np_random.random(OBS_DIM).astype(np.float32), "action_mask": mask}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        reward = 1.0 if self._mask[int(action)] else -1.0
        self._t += 1
        terminated = bool(self.np_random.random() < 0.05)
        truncated = self._t >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=SpikeEnv)


def make_env(seed: int = 0) -> TransformedEnv:
    env = TransformedEnv(
        GymEnv(ENV_ID, categorical_action_encoding=True),
        Compose(
            DTypeCastTransform(torch.int8, torch.bool, in_keys=["action_mask"]),
            ActionMask(),
        ),
    )
    env.set_seed(seed)
    return env


def assert_actions_valid(td):
    actions = td["action"].reshape(-1)
    masks = td["action_mask"].reshape(-1, N_ACTIONS)
    assert masks.dtype == torch.bool
    assert masks[torch.arange(actions.numel()), actions].all(), "an action outside the mask was taken"


def make_ppo_modules(env):
    actor_net = TensorDictModule(MLP(in_features=OBS_DIM, out_features=N_ACTIONS, num_cells=[32]), ["observation"], ["logits"])
    actor = ProbabilisticActor(
        actor_net,
        in_keys={"logits": "logits", "mask": "action_mask"},
        distribution_class=MaskedCategorical,
        spec=env.action_spec,
        return_log_prob=True,
    )
    critic = ValueOperator(MLP(in_features=OBS_DIM, out_features=1, num_cells=[32]), in_keys=["observation"])
    return actor, critic


def test_env_specs_and_types():
    env = make_env()
    check_env_specs(env)
    td = env.reset()
    assert td["observation"].shape == (OBS_DIM,)
    assert td["action_mask"].dtype == torch.bool
    assert env.action_spec.shape == ()  # categorical (integer) actions, not one-hot


def test_random_rollout_respects_mask():
    env = make_env()
    td = env.rollout(200, break_when_any_done=False)
    assert td.batch_size == (200,)
    assert_actions_valid(td)


@pytest.fixture(scope="module")
def ppo_batch():
    torch.manual_seed(0)
    env = make_env()
    actor, critic = make_ppo_modules(env)
    collector = Collector(env, actor, frames_per_batch=128, total_frames=128, auto_register_policy_transforms=False)
    batch = next(iter(collector))
    collector.shutdown()
    return actor, critic, batch


def test_collector_batch_with_masked_actor(ppo_batch):
    _, _, batch = ppo_batch
    assert batch.numel() == 128
    for key in ("observation", "action_mask", "action", "action_log_prob", ("next", "reward")):
        assert key in batch.keys(include_nested=True)
    assert_actions_valid(batch)


def _step_loss(loss_module, td, loss_keys):
    optim = torch.optim.Adam(loss_module.parameters(), lr=1e-3)
    out = loss_module(td)
    total = sum(out[k] for k in loss_keys)
    assert torch.isfinite(total), f"non-finite loss: {out}"
    optim.zero_grad()
    total.backward()
    grads = [p.grad for p in loss_module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    optim.step()
    return out


def test_ppo_update(ppo_batch):
    actor, critic, batch = ppo_batch
    td = batch.clone()
    GAE(gamma=0.99, lmbda=0.95, value_network=critic)(td)
    loss = ClipPPOLoss(actor, critic, clip_epsilon=0.2, entropy_coeff=0.01)
    out = _step_loss(loss, td, ["loss_objective", "loss_critic", "loss_entropy"])
    assert torch.isfinite(out["entropy"])  # needed for the policy-entropy figure


def test_a2c_update(ppo_batch):
    actor, critic, batch = ppo_batch
    td = batch.clone()
    GAE(gamma=0.99, lmbda=1.0, value_network=critic)(td)
    _step_loss(A2CLoss(actor, critic, entropy_coeff=0.01), td, ["loss_objective", "loss_critic", "loss_entropy"])


def test_reinforce_update(ppo_batch):
    actor, critic, batch = ppo_batch
    td = batch.clone()
    GAE(gamma=0.99, lmbda=1.0, value_network=critic)(td)
    _step_loss(ReinforceLoss(actor, critic), td, ["loss_actor", "loss_value"])


def test_dqn_masked_update():
    torch.manual_seed(0)
    env = make_env()
    qnet = QValueActor(
        MLP(in_features=OBS_DIM, out_features=N_ACTIONS, num_cells=[32]),
        in_keys=["observation"],
        spec=env.action_spec,
        action_mask_key="action_mask",
    )
    # Greedy actions must respect the mask on their own, before exploration is added.
    assert_actions_valid(qnet(env.rollout(50, break_when_any_done=False)))

    explore = EGreedyModule(env.action_spec, eps_init=1.0, eps_end=0.1, annealing_num_steps=1000, action_mask_key="action_mask")
    policy = TensorDictSequential(qnet, explore)
    collector = Collector(env, policy, frames_per_batch=128, total_frames=128, auto_register_policy_transforms=False)
    batch = next(iter(collector))
    collector.shutdown()
    assert_actions_valid(batch)

    rb = ReplayBuffer(storage=LazyTensorStorage(1000), batch_size=64)
    rb.extend(batch.reshape(-1))
    loss = DQNLoss(qnet, action_space=env.action_spec, delay_value=True, double_dqn=True)
    loss.make_value_estimator(gamma=0.99)
    target_updater = HardUpdate(loss, value_network_update_interval=10)
    out = _step_loss(loss, rb.sample(), ["loss"])
    target_updater.step()
    assert torch.isfinite(out["loss"])
