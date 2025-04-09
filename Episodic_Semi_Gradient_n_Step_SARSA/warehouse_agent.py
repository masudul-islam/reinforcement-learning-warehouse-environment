import numpy as np
from itertools import product

class warehouse_agent:
    def __init__(self, rows, columns):
        # Action = ['Up' 0, 'Right' 1, 'Down' 2, 'Left' 3], clockwise
        self.A = np.array([0, 1, 2, 3])
        # Transition function
        self.P = np.array([0.7, 0.82, 0.94, 1])
        self.reward = None
        self.terminal_states = None
        self.policies = None
        self.q_values = None
        self.num_aisles = None
        self.columns = columns
        self.rows = rows
        self.H = None  # environment height
        self.W = None  # environment width
        self.num_terminal_states = rows * columns
    # def generate_environment(self, rows, columns)
        # initial gridworld
        H = 3 * rows + rows + 1
        W = 3 * columns + 2
        reward = np.array([[i, j] for i in range(H) for j in range(W)])

        reward = -np.ones((H, W))
        reward[:, np.arange(1, W-3, 3)] = -9
        reward[:, np.arange(3, W, 3)] = -9
        reward[np.arange(0, H, 4)] = -1
        self.reward = reward
        self.num_aisles = rows * columns
        # number of aisles, number of states, number of actions
        self.q_values = np.random.rand(self.num_aisles, H*W, 4)
        terminal_states_x = np.arange(2, H, 4)
        terminal_states_y = np.arange(2, W, 3)  
        self.terminal_states = np.array(list(product(terminal_states_x, terminal_states_y)))
        self.H = H
        self.W = W

    # generate item that need to be delivered, sort terminal states the agent need to visit based on their distance to [0, 0] where the agent start,
    # return sorted terminal states and their sequency
    def generate_items(self, num_items):
        if num_items > self.num_aisles:
            print(f'bad number items, make it <= {self.num_aisles}')
            return None
        random_numbers = np.random.choice(np.arange(0, self.num_aisles, 1), size=num_items, replace=False)
        items_terminal_states = self.terminal_states[random_numbers]
        sequence = np.argsort(np.linalg.norm(items_terminal_states, axis=1))
        return items_terminal_states[sequence], np.sort(random_numbers)

    # def compute_state_value(self, row: int, column: int, action: int, gamma: float, state_values: np.ndarray) -> int:
    #     value = 0
    #
    #     for i in range(4):
    #         if i == 0:
    #             new_row, new_column = compute_next_position(row, column, ACTION[action])  # specified direction
    #         elif i == 1:
    #             new_row, new_column = compute_next_position(row, column, direction_change(action,
    #                                                                                       'Right'))  # Right of specified direction
    #         elif i == 2:
    #             new_row, new_column = compute_next_position(row, column, direction_change(action,
    #                                                                                       'Left'))  # Left of specified direction
    #         else:
    #             new_row, new_column = row, column  # remain same position
    #         value += TRANSITION_PROBS[i] * (STATE[new_row][new_column] + gamma * state_values[new_row, new_column])
    #     return value
    #
    # # compute updated state_values, state_actions, number of iteration
    # def value_iteration_inplace(self, gamma: float, catnip_terminal: bool):
    #     state_values = np.random.random(self.reward.shape)
    #     state_actions = [['None' for _ in range(5)] for _ in range(5)]
    #     iteration = 0
    #     terminate_reward = 10
    #     terminate_states = self.terminal_states[0]
    #     while True:
    #         difference = 0
    #         iteration += 1
    #         # temp_values = state_values.copy()  # temporary value
    #         for i in range(5):
    #             for j in range(5):
    #                 if self.reward[i, j] == 'Wall' or self.reward[i, j] == terminate_reward :
    #                     continue
    #                 max_state_value = float('-inf')
    #                 # find action that give maximum value
    #                 for a in range(4):
    #                     s_value = compute_state_value(i, j, action=a, gamma=gamma,
    #                                                   state_values=state_values)  # pass temporary values
    #                     if s_value > max_state_value:
    #                         max_state_value = s_value
    #                         # state_actions[i][j] = ACTION[a]
    #                 # compute difference of new state value and current state value
    #                 state_value_diff = np.abs(max_state_value - state_values[i, j])
    #                 difference = np.maximum(difference, state_value_diff)
    #                 # update difference
    #                 state_values[i, j] = max_state_value
    #
    #         if difference < 0.0001:
    #             break
    #     # update the policy
    #     for i in range(5):
    #         for j in range(5):
    #             if self.reward[i, j] == 'Wall' or self.reward[i, j] == 10:
    #                 continue
    #             if i == 0 and j == 1 and catnip_terminal:
    #                 continue
    #             max_state_value = float('-inf')
    #             for a in range(4):
    #                 s_value = compute_state_value(i, j, action=a, gamma=gamma,
    #                                               state_values=state_values)  # pass temporary values
    #                 if s_value > max_state_value:
    #                     max_state_value = s_value
    #                     state_actions[i][j] = self.A[0]  # record state
    #     return np.round(state_values, 4)
    #

if __name__ == '__main__':
    agent = warehouse_agent(2, 2)
    print(agent.reward)

