import datetime
import pathlib
import time
import numpy
import torch
import random
from copy import deepcopy
from .abstract_game import AbstractGame

grid_size = 10
seed = numpy.random.randint(100000)
class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        # 下面 无论是否使用GPU 都不要改 None是默认使用全部GPU  如果没有GPU 自动使用CPU
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        # 9 是因为 3x3 在 GridEnv的get_observation中被拉成1维度的9
        #self.observation_shape = (1, 1, 9)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        #self.observation_shape = (1, 1, grid_size*grid_size)
        #self.observation_shape = (1,grid_size, grid_size)
        # grid和marked_position 两个np.array 所以是2 。这次不在grid上修改保留原始信息
        self.observation_shape = (2,grid_size, grid_size)
        self.action_space = list(range(grid_size*grid_size))#list(range(2))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1#2#1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        # 要使用GPU 必须环境中有GPU  这里再改成True
        # https://github.com/ray-project/ray/issues/30012#issuecomment-1364633366
        # pip install grpcio==1.51.3 就可以正常使用gpu了  还有说法是ray==2.0.0
        
        self.selfplay_on_gpu = False#True #False
        self.max_moves = grid_size//2#6  # Maximum number of moves if game is not finished before
        self.num_simulations = 100 # Number of future moves self-simulated
        self.discount = 1# 0.978  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3#0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"#"fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 1#10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6#1  # Number of blocks in the ResNet
        self.channels = 128#2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2#2  # Number of channels in reward head
        self.reduced_channels_value = 2#2  # Number of channels in value head
        self.reduced_channels_policy = 4#2  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32#5
        self.fc_representation_layers =[] #[16]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64] #[16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64] #[16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [] #[16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [] #[16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 2000#30000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size =  32  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 50#10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25#0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 1e-3#0.0064  # Initial learning rate
        self.lr_decay_rate = 0.95#1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000#1000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 5  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0.2  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        #return 0        
        if trained_steps < 5000:
            return 1
        elif trained_steps < 30000:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        # modified 20240226
        self.env = GridEnv(size=grid_size)
        print(f'h_score={self.env.h_score}')

    def step(self, action):
        """
        Apply action to the game.


        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        #return [[observation]], reward * 10, done
        #print([[observation]])
        #return [[observation]], reward , done
        #return [observation], reward , done
        return observation, reward , done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        #return list(range(2))
        # 直接使用list(range(2)) 即down和right太粗暴 没有考虑撞墙的情况
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        #return [[self.env.reset()]]
        #return [self.env.reset()]
        return self.env.reset()


    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        #https://blog.csdn.net/qq_19446965/article/details/126793548
        # input接收用户输入 的目的是中断让render按照顺序输出各个observation 那么此时用time.sleep也能实现这个目的
        time.sleep(0.2)
        #input("Press enter to take a step .按下enter让agent执行一步.这个步骤的作用主要是让进程停顿 从而让并发的输出有一定的顺序.")
        #print("Press enter to take a step .目前自动按下enter.本提示来自game目录的对应游戏的py文件中的def render方法.")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        # actions = {
        #     0: "Down",
        #     1: "Right",
        # }
        actions ={k:str([k//grid_size , k %grid_size]) for k in range(0,grid_size*grid_size)}
        return f"{action_number}. action= {actions[action_number]}"


class GridEnv:
    def __init__(self, size=3):
        self.size = size
        
        self.MARK_NEGATIVE = -10.0
        
        self.agent_get_reward =0
        # 原始的action space为[0,100)
        
        # 每次step都会更新 _used_actions ，使用_actions - _used_actions - _invalid_actions，剩下的才是合法的action space
        
        # position reset
        self.position = None # [0, 0]
        
        # grid reset
        #a_100 = list(range(1, grid_size*grid_size + 1))
        #random.shuffle(a_100)
        #self.grid = numpy.array(a_100).reshape(grid_size, grid_size) / len(a_100)  # np.random.random((10, 10))
        numpy.random.seed(seed)
        self.grid = numpy.random.rand(grid_size,grid_size)
        numpy.fill_diagonal(self.grid, self.MARK_NEGATIVE)
        # marked_position rest
        self.mark = numpy.zeros([grid_size,grid_size])
        # h score reset 
        self.h_score = self.heuristic_score()
        print(f'h_score={self.h_score}')
        self.agent_get_reward =0
        # 每次step都会更新 _used_actions ，使用_actions - _used_actions - _invalid_actions，剩下的才是合法的action space
        self._used_actions=set([])
        # invalid actions 比如0 11,22,,,99
        self._invalid_actions = set([i for i in range(grid_size*grid_size) if i//grid_size == i%grid_size])
        # action space reset
        self._actions = set(range(grid_size * grid_size)) -self._invalid_actions
        
        
    # def legal_actions(self):
    #    legal_actions = list(range(2))
    #    if self.position[0] == (self.size - 1):
    #        legal_actions.remove(0)
    #    if self.position[1] == (self.size - 1):
    #        legal_actions.remove(1)
    #    return legal_actions
    
    def heuristic_score(self):
        heuristic_score = 0.0
        grid_copy = self.grid.copy()
        while numpy.max(grid_copy)> self.MARK_NEGATIVE/2.0 :
            #print(grid_copy)
            #print(np.max(grid_copy))
            heuristic_score += numpy.max(grid_copy)
            m = numpy.argmax(grid_copy)                # 把矩阵拉成一维，m是在一维数组中最大值的下标
            row, col = divmod(m, grid_copy.shape[1])    # r和c分别为商和余数，即最大值在矩阵中的行和列 # m是被除数， a.shape[1]是除数
            #print(row, col)
            grid_copy[[row,col],:]=self.MARK_NEGATIVE 
            grid_copy[:,[row,col]]=self.MARK_NEGATIVE
            #print(grid)
        #print(f'heuristic_score ={heuristic_score}')
        #print(f'h_s={heuristic_scores}')
        #assert False
        return heuristic_score
    
    
    def legal_actions(self):
        legal_actions = self._actions
        if self.position and len(self.position)>1:
            # for example self.position=[2,9]
            #chosen_action = self.position[0]*grid_size + self.position[1]
            marked_row_act_0 = set(range(self.position[0]*grid_size,(self.position[0]+1)*grid_size))
            marked_row_act_1 = set(range(self.position[1] * grid_size, (self.position[1] + 1) * grid_size))
            marked_col_act_0 = set([idx*grid_size + self.position[0] for idx in range(grid_size)])
            marked_col_act_1 = set([idx * grid_size + self.position[1] for idx in range(grid_size)])
            self._used_actions = self._used_actions | marked_row_act_0 | marked_row_act_1 | marked_col_act_0 | marked_col_act_1
            legal_actions = list(legal_actions -self._invalid_actions -  self._used_actions)
        #print(f'legal_actions={legal_actions}')
        return legal_actions #list(self._actions)
        
        
    # def step(self, action):
    #     if action not in self.legal_actions():
    #         pass
    #     elif action == 0:
    #         self.position[0] += 1
    #     elif action == 1:
    #         self.position[1] += 1
    #
    #     reward = 1 if self.position == [self.size - 1] * 2 else 0
    #     return self.get_observation(), reward, bool(reward)

    def step(self, action):
        #print(f'step legal actions={self.legal_actions()}')
        if action not in self.legal_actions() or len(self.legal_actions())==0 :
            pass
        if not self.position:
            self.position =[-1,-1] # position[-1,-1]表示不在grid上的位置只是为了占位
        self.position[0] = action // grid_size
        self.position[1] = action %  grid_size
        #reward = 1 if self.position == [self.size - 1] * 2 else 0
        #不能写成reward = self.grid[self.position] 因为self.position=[1,1] 会导致grid[1,1]取得是两行
        # 或者写成reward = self.grid[self.position[0],self.position[1]] 
        #reward = self.grid[*self.position] 
        reward = self.grid[self.position[0],self.position[1]]  - self.mark[self.position[0],self.position[1]] #- self.h_score / (grid_size/2)
        self.agent_get_reward += reward
        #print(f'123reward={reward}')
        # grid 变化太剧烈? 所以换成mark来记录已经不能下的位置
        self.grid[self.position, :] = self.MARK_NEGATIVE
        self.grid[:, self.position] = self.MARK_NEGATIVE
        self.mark[self.position, :] = self.MARK_NEGATIVE
        self.mark[:, self.position] = self.MARK_NEGATIVE
        #done = (numpy.max(self.grid) <= self.MARK_NEGATIVE) or len(self.legal_actions())==0
        done = (numpy.max(self.mark) <= self.MARK_NEGATIVE) or len(self.legal_actions())==0
        #done =  len(self.legal_actions())==0
        #reward =0
        #if done :
        #    #if self.agent_get_reward>= self.h_score :
        #    reward = self.agent_get_reward - self.h_score
            
        
        return self.get_observation(), reward, done#bool(reward)

    def reset(self):
        # position reset
        self.position = None # [0, 0]
        
        # grid reset
        #a_100 = list(range(1, grid_size*grid_size + 1))
        #random.shuffle(a_100)
        #self.grid = numpy.array(a_100).reshape(grid_size, grid_size) / len(a_100)  # np.random.random((10, 10))
        numpy.random.seed(seed)
        #self.grid = numpy.random.rand(grid_size,grid_size)
        numpy.fill_diagonal(self.grid, self.MARK_NEGATIVE)

        # marked_position reset
        self.mark = numpy.zeros([grid_size,grid_size])
        
        # h score reset 
        self.h_score = self.heuristic_score()
        self.agent_get_reward =0
        # 每次step都会更新 _used_actions ，使用_actions - _used_actions - _invalid_actions，剩下的才是合法的action space
        self._used_actions=set([])
        # invalid actions 比如0 11,22,,,99
        self._invalid_actions = set([i for i in range(grid_size*grid_size) if i//grid_size == i%grid_size])
        # action space reset
        self._actions = set(range(grid_size * grid_size)) -self._invalid_actions
        return self.get_observation()

    def render(self):
        #im = numpy.full((self.size, self.size), "-")
        #im[self.size - 1, self.size - 1] = "1"
        im = deepcopy(self.grid) - deepcopy(self.mark)
        if self.position and len(self.position)>0:
            im[self.position[0], self.position[1]] = 200
        print(im)

    def get_observation(self):
        #observation = numpy.zeros((self.size, self.size))
        #observation[self.position[0]][self.position[1]] = 1
        observation = [self.grid ,self.mark]
        # flatten 把二维3x3 拉成 单独的1维为9的np array
        #return observation.flatten()
        return numpy.array(observation)
