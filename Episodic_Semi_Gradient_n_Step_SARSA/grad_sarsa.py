# Episodic semi-gradient n-step Sarsa
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from warehouse_agent import warehouse_agent
from cat_vs_monster_domain import *
from q_network import *
DTYPE = torch.float32


class SemiGradNStepSarsa(warehouse_agent):

    def __init__(self, rows, columns):
        super().__init__(rows, columns)
        self.len_action = len(self.A)
        self.q_net = []
        self.num_episodes = 4000
        for _ in range(self.num_terminal_states):
            self.q_net.append(QNetwork())

    def generate_ini_state(self, terminate_x, terminate_y):
        while True:
            x = int(np.random.uniform(0, self.H))
            y = int(np.random.uniform(0, self.W))
            if x != terminate_x or y != terminate_y:
                return x, y

    def estimate_q_value(self, aisle, x, y, a):
        action_one_hot = torch.zeros(self.len_action).detach()
        action_one_hot[int(a)] = 1.0
        state = torch.tensor([x, y]).detach()
        input_tensor = torch.cat([state, action_one_hot], dim=0)
        q_value = self.q_net[aisle](input_tensor)
        return q_value

    def estimate_q_values(self, aisle, x, y):
        state = torch.tensor([x, y]).repeat(self.len_action, 1)
        actions = torch.eye(self.len_action)
        input_tensor = torch.cat([state, actions], dim=1)
        q_values = self.q_net[aisle](input_tensor).view(1, -1)
        return q_values[0]


    def apply_transition_function(self, intend_action):
        c = np.random.uniform(0, 1)
        if c < self.P[0]:  # move intended direction
            return intend_action
        elif c < self.P[1]:  # turn Left
            return self.A[intend_action - 1]
        elif c < self.P[2]:  # turn Right

            return self.A[(intend_action + 1) % self.len_action]
        else:
            return None

    def generateAction_epsGreedy(self, aisle_num, x, y, eps):
        q_values = self.estimate_q_values(aisle_num, x, y)

        # Find best action(s)
        max_q_value = torch.max(q_values)
        best_action_positions = (q_values == max_q_value).nonzero(as_tuple=True)[0]
        num_best_actions = best_action_positions.shape[0]

        # Compute action probabilities
        pi = np.full(self.len_action, eps / self.len_action)
        pi[best_action_positions] += (1 - eps) / num_best_actions
        # Sample an action based on computed probabilities
        intend_action = np.random.choice(self.len_action, p=pi)

        real_action = self.apply_transition_function(intend_action)
        return intend_action, real_action


    def computeNextPosition(self, row: int, column: int, action) -> tuple[int, int]:
        if action is None:  # no action, stay on the same state
            return row, column
        new_row = row
        new_column = column
        if action == 0:
            new_row = row - 1
        elif action == 1:
            new_column = column + 1
        elif action == 2:
            new_row = row + 1
        elif action == 3:
            new_column = column - 1
        # move outside of map
        if new_row < 0 or new_row > 4 or new_column < 0 or new_column > 4:
            return row, column
        # # hit wall
        # elif REWARD[new_row][new_column] == 'Wall':
        #     return row, column
        # regular move or stay same place
        else:  # action == 'None'
            return new_row, new_column


    def computeG(self, history, gamma, i):
        G = 0
        for Rt in range(i):
            G += (gamma ** Rt) * history[Rt][0]
        return G


    def set_reward(self, reward, terminate_x, terminate_y):
        reward_map = self.reward - np.array([terminate_x, terminate_y])
        reward_map = -np.linalg.norm(reward_map, axis=1).reshape(self.H, self.W) * 0.5
        a = reward_map[np.arange(0, self.H, 4)]
        # set the shelves reward
        reward_map[:, np.arange(1, self.W-3, 3)] = -9
        reward_map[:, np.arange(3, self.W, 3)] = -9
        reward_map[np.arange(0, self.H, 4)] = a
        # set ternimate state reward
        reward_map[terminate_x, terminate_y] = reward
        return reward_map


    def train(self, n: int, alpha: float, gamma: float, epsilon: float):
        steps = []
        loss_fn = torch.nn.MSELoss()
        for aisle in range(len(self.terminal_states)):  # multiple terminate states

        # for aisle in range(1):  # multiple terminate states
            if aisle != 0:  # set previous trained parameter values to new network
                if aisle % self.columns == 0:
                    self.q_net[aisle].load_state_dict(self.q_net[aisle - self.columns].state_dict())
                    # self.q_net[aisle].load_state_dict(self.q_net[aisle - self.columns].state_dict())
                    print(f"Parameters of QNetwork[{aisle - self.columns}] copied to QNetwork[{aisle}].")
                else:
                    self.q_net[aisle].load_state_dict(self.q_net[aisle - 1].state_dict())
                    print(f"Parameters of QNetwork[{aisle - 1}] copied to QNetwork[{aisle}].")
            # ################################
            # model_path = "base_q_net_params_0_2by1_3.pth"
            # self.q_net[aisle].load_state_dict(torch.load(model_path, weights_only=True))
            ################################
            terminate_x, terminate_y = self.terminal_states[aisle]
            optimizer = torch.optim.Adam(self.q_net[aisle].parameters(), lr=alpha)
            temp_reward = self.reward[terminate_x, terminate_y]
            self.reward[terminate_x, terminate_y] =4
            #self.reward = self.set_reward(6, terminate_x, terminate_y)  # set reward
            for episode in range(self.num_episodes):
                if episode % 100 == 0:
                    print(f"aisle: {aisle}, Episode: {episode}")
                stored_n_step = []  # len should equal 4n, [[x_0, y_0, A_0], [R_1, x_1, y_1, A_1], [R_2, ...]]
                x, y = self.generate_ini_state(terminate_x, terminate_y)
                intend_action, real_action = self.generateAction_epsGreedy(aisle, x, y, epsilon)
                stored_n_step.append(
                    [None, torch.tensor(x, dtype=DTYPE).item(), torch.tensor(y, dtype=DTYPE).item(), intend_action])
                T = np.inf
                t = 0
                tau = None
                while True:
                    if t < T:
                        x, y = self.computeNextPosition(x, y, real_action)
                        stored_n_step.append([self.reward[x, y],
                                              torch.tensor(x, dtype=DTYPE).item(), torch.tensor(y, dtype=DTYPE).item()])
                    if (x == terminate_x and y == terminate_y): # or self.reward[x, y] == -9:
                        T = t + 1
                        x, y = -1, -1
                    else:
                        intend_action, real_action = self.generateAction_epsGreedy(aisle, x, y, epsilon)
                        stored_n_step[-1].append(intend_action)

                    # print(f't: {t}')
                    # print(stored_n_step)
                    tau = t - n + 1
                    if tau >= 0:
                        i = min(tau + n, T)
                        G = self.computeG(stored_n_step[1:], gamma, len(stored_n_step) - 1)
                        if tau + n < T:
                            G += (gamma ** n) * self.estimate_q_value(aisle, *stored_n_step[-1][1:]).detach()

                        action_one_hot = torch.zeros(self.len_action, dtype=DTYPE)
                        action_one_hot[stored_n_step[0][3]] = 1.0
                        state = torch.tensor([stored_n_step[0][1], stored_n_step[0][2]], dtype=DTYPE)
                        input_tensor = torch.cat([state, action_one_hot], dim=0)

                        q_value_tau_step = self.q_net[aisle](input_tensor)
                        # Compute loss
                        loss = loss_fn(q_value_tau_step, torch.tensor([G], dtype=DTYPE))
                        # Backpropagation and optimization step
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # q_value_tau_step = self.q_net(input_tensor)
                        # q_value_tau_step.backward()
                        #
                        # error = G - q_value_tau_step
                        # # Manually update weights based on the TD error and learning rate
                        # with torch.no_grad():
                        #     for param in self.q_net.parameters():
                        #         param += alpha * error * param.grad
                        # self.q_net.zero_grad()
                        stored_n_step = stored_n_step[1:]

                    t += 1
                    if tau == T - 1 or t > 1000:
                        print(f't is {t}')
                        steps.append(t)
                        break
            self.reward[terminate_x, terminate_y] = temp_reward
        return np.array(steps)

    def present_policy(self):
        for aisle in range(len(self.q_net)):
            policy = torch.zeros(self.H, self.W)
            for i in range(policy.shape[0]):
                for j in range(policy.shape[1]):
                    actions = self.estimate_q_values(aisle, i, j)
                    argmax = torch.argmax(actions)
                    policy[i, j] = argmax
            policy[self.terminal_states[aisle][0], self.terminal_states[aisle][1]] = -1
            # aa = policy[np.arange(0, self.W, 4)]
            # policy[:, np.arange(1, self.W - 3, 3)] = float('nan')
            # policy[:, np.arange(3, self.W, 3)] = float('nan')
            # policy[np.arange(0, self.W, 4)] = aa
            p = np.zeros((self.H, self.W), dtype=str)
            for i in range(self.H):
                for j in range(self.W):
                    if policy[i, j] == 1:
                        p[i, j] = '\u2192'
                    elif policy[i, j] == 3:
                        p[i, j] = '\u2190'
                    elif policy[i, j]== 0:
                        p[i, j] = '\u2191'
                    elif policy[i, j] == 2:
                        p[i, j] = '\u2193'
                    elif policy[i, j] == -1:
                        p[i, j] = ' '
            print(np.array(p))

    def n_step_analysis(self):
        plt.figure()
        for n in [1, 3, 7]:
            avg_steps = []
            for _ in range(20):
                if len(avg_steps) == 0:
                    avg_steps = self.train(n, alpha=1e-4, gamma=0.92, epsilon=0.1)
                else:
                    avg_steps += self.train(n, alpha=1e-4, gamma=0.92, epsilon=0.1)
            plt.plot(np.arange(0, self.num_episodes), avg_steps / 20, marker='.', label=f"n = {n}")
        plt.xlabel("episodes")
        plt.ylabel("avg steps")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    ro = 2
    co = 2
    agent = SemiGradNStepSarsa(ro, co)
    # load parameters
    # agent.q_net[0].load_state_dict(torch.load("q_net_base.pth", weights_only=True))
    # print(agent.present_policy())

    alpha = 1e-4
    gamma = 0.92
    epsilon = 0.1
    n = 3
    agent.num_episodes = 4000
    #agent.n_step_analysis()

    start_time = time.time()
    agent.train(n, alpha, gamma, epsilon)
    end_time = time.time()  # Record the end time

    elapsed_time = (end_time - start_time) / 3600
    print(f"Training completed in {elapsed_time:.2f} hours.")

    print(agent.present_policy())
    for aisle in range(len(agent.q_net)):
        torch.save(agent.q_net[aisle].state_dict(), f"detach_q_net_params_{aisle}_{ro}by{co}_{n}.pth")
    print("Q-network parameters saved.")
