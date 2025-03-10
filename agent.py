
from abc import abstractmethod

class Agent:
    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def save_models(self, episode, path=""):
        pass

    @abstractmethod
    def load_models(self, path, second_path):
        pass

    @abstractmethod
    def learn(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def learn(self, states, actions, rewards, next_states, dones, masks):
        pass
    
    @abstractmethod
    def decrease_epsilon(self):
        pass

    @abstractmethod
    def update_target(self):
        pass