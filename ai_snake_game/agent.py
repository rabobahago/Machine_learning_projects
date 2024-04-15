import numpy as np
import random
import torch
from collections import deque
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)
    def get_game(self,game):
        pass
    def remember(self, state, action, reward, next_state, done):
        pass
    def train_long_memory(self):
        pass
    def train_short_memory(self, state, action, reward, next_state, done):
        pass
    def get_action(self, state):
        pass
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()
            if score > record:
                record = score
            print('Game: ', agent.n_game, 'Score: ', score, 'Record: ', record)

if __name__ == '__main__':
    train()