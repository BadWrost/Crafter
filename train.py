import argparse
import pickle
from pathlib import Path
from model import DQN
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from src.crafter_wrapper import Env
import matplotlib.pyplot as plt
from agent import DQNAgent, Transition, RandomAgent



def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval(agent,policy_net , env, crt_step, opt):
    """ Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during training.
    """
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        obs = obs[-1].unsqueeze(0).unsqueeze(0)
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs,policy_net,force_policy=True)
            obs, reward, done, info = env.step(action)
            obs = obs[-1].unsqueeze(0).unsqueeze(0)
            episodic_returns[-1] += reward

    _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},64,64),"
        + "with values between 0 and 1."
    )

def train_agent(agent, policy_net, target_net, optimizer, batch_size, GAMMA, TARGET_UPDATE, ep_cnt):
    transitions = agent.get_memories(batch_size)
    # Convert memories to tensors
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device='cpu', dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    # Concatenate the actions using numpy
    
    tmp_action = []
    for action in batch.action:
        action = torch.tensor([[action]], device='cpu')
        tmp_action.append(action)
    action_batch = torch.cat(tmp_action)
    reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device='cpu')
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch


    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # Update the target network, copying all weights and biases in DQN
    if ep_cnt % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


def main(opt):
    _info(opt)
    #opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = torch.device("cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    obs = env.reset()
    current_screen = obs[-1].unsqueeze(0).unsqueeze(0)
    last_screen = obs[-2].unsqueeze(0).unsqueeze(0)
    _, screen_height, screen_width = obs.shape
    # plot the screen
    plt.figure()
    plt.imshow(current_screen[0].permute(1, 2, 0))
    plt.title('Example extracted screen '+str(current_screen.shape))
    plt.show()

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    policy_net = DQN(screen_height, screen_width, n_actions).to("cpu")
    target_net = DQN(screen_height, screen_width, n_actions).to("cpu")
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())

    agent = DQNAgent(env.action_space.n, EPS_END, EPS_START, EPS_DECAY)

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    
    state = current_screen - last_screen
    while step_cnt < opt.steps or not done:
        print(f"\rStep {step_cnt}", end="")
        if done:
            print(f"\nEpisode {ep_cnt}\n", end="")
            ep_cnt += 1
            obs, done = env.reset(), False
            current_screen = obs[-1].unsqueeze(0).unsqueeze(0)
            last_screen = obs[-2].unsqueeze(0).unsqueeze(0)
            state = current_screen - last_screen
            next_state = None

        action = agent.act(state, policy_net, step_cnt)
        obs, reward, done, info = env.step(action)
        reward = torch.tensor([reward], device='cpu')
        
        last_screen = current_screen
        current_screen = obs[-1].unsqueeze(0).unsqueeze(0)
        next_state = current_screen - last_screen
        

        step_cnt += 1
        agent.add_memory(state, action, next_state, reward)
        state = next_state
        
        if len(agent) >= BATCH_SIZE:
            train_agent(agent, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA, TARGET_UPDATE, ep_cnt)
        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            print("Evaluating...")
            eval(agent,policy_net, eval_env, step_cnt, opt)


def get_options():
    """ Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/DQNAgent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_options())
