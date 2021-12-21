from abc import ABC, abstractmethod

class Memory(ABC):
    def __init__(self):
        self.data_keys = ['states', 'actions', 'next_states', 'priorities']

    @abstractmethod
    def reset(self):
        '''Method to fully reset the memory storage and related variables'''
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        '''
        Implement memory update given the full info from the latest timestep. NOTE: guard for np.nan reward and done when individual env resets.
        Return True if memory is ready to be sampled for training, False  otherwise
        '''
        pass

    @abstractmethod
    def sample(self):
        '''Implement memory sampling mechanism'''
        pass


