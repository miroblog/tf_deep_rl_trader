# Deep RL Trader + PPO Agent Implemented using Tensorforce

This repo contains 
1. Trading environment(OpenAI Gym) + Wrapper for Tensorforce Env 
2. PPO(Proximal Policy Optimization) Agent (https://arxiv.org/abs/1707.06347)
Agent is implemented using `tensorforce`(https://github.com/reinforceio/tensorforce)     
  
Agent is expected to learn useful action sequences to maximize profit in a given environment.  
Environment limits agent to either buy, sell, hold stock(coin) at each step.  
If an agent decides to take a   
* LONG position it will initiate sequence of action such as `buy- hold- hold- sell`    
* for a SHORT position vice versa (e.g.) `sell - hold -hold -buy`.    

Only a single position can be opened per trade. 
* Thus invalid action sequence like `buy - buy` will be considered `buy- hold`.   
* Default transaction fee is : 0.0005  

Reward is given
* when the position is closed or
* an episode is finished.   
  
This type of sparse reward granting scheme takes longer to train but is most successful at learning long term dependencies.  

Agent decides optimal action by observing its environment.  
* Trading environment will emit features derived from ohlcv-candles(the window size can be configured). 
* Thus, input given to the agent is of the shape `(window_size, n_features)`.  

With some modification it can easily be applied to stocks, futures or foregin exchange as well.

[Visualization](https://github.com/miroblog/tf_deep_rl_trader/blob/master/visualize_info.ipynb) / [Main](https://github.com/miroblog/tf_deep_rl_trader/blob/master/ppo_trader.py) / [Environment](https://github.com/miroblog/tf_deep_rl_trader/blob/master/env/TFTraderEnv.py)

Sample data provided is 5min ohlcv candle fetched from bitmex.
* train : `'./data/train/` 70000
* test : `'./data/train/` 16000

### Prerequisites

keras-rl, numpy, tensorflow ... etc

```python
pip install -r requirements.txt

```

## Getting Started 

### Create Environment & Agent
```python
# create environment
# OPTIONS
# create environment for train and test
PATH_TRAIN = "./data/train/"
PATH_TEST = "./data/test/"
TIMESTEP = 30  # window size
environment = create_btc_env(window_size=TIMESTEP, path=PATH_TRAIN, train=True)
test_environment = create_btc_env(window_size=TIMESTEP, path=PATH_TEST, train=False)

# create spec for network and baseline
network_spec = create_network_spec() # json format
baseline_spec = create_baseline_spec()

# create agent
agent = PPOAgent(
    discount=0.9999,
    states=environment.states,
    actions=environment.actions,
    network=network_spec,
    # Agent
    states_preprocessing=None,
    actions_exploration=None,
    reward_preprocessing=None,
    # MemoryModel
    update_mode=dict(
        unit='timesteps',  # 'episodes',
        # 10 episodes per update
        batch_size=32,
        # # Every 10 episodes
        frequency=10
    ),
    memory=dict(
        type='latest',
        include_next_states=False,
        capacity=50000
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=0.0,  # None
    # PGModel

    baseline_mode='states',
    baseline=dict(type='custom', network=baseline_spec),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=(1e-4)  # 3e-4
        ),
        num_steps=5
    ),
    gae_lambda=0,  # 0
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=(1e-4)  # 1e-4
    ),
    subsampling_fraction=0.2,  # 0.1
    optimization_steps=10,
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    )
)

```

### Train and Validate
```python
    train_runner = Runner(agent=agent, environment=environment)
    test_runner = Runner(
        agent=agent,
        environment=test_environment,
    )

    train_runner.run(episodes=100, max_episode_timesteps=16000, episode_finished=episode_finished)
    print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
        ep=train_runner.episode,
        ar=np.mean(train_runner.episode_rewards[-100:]))
    )

    test_runner.run(num_episodes=1, deterministic=True, testing=True, episode_finished=print_simple_log)
```

### Configuring Agent
```python
## you can stack layers using blocks provided by tensorforce or define ur own...
def create_network_spec():
    network_spec = [
        {
            "type": "flatten"
        },
        dict(type='dense', size=32, activation='relu'),
        dict(type='dense', size=32, activation='relu'),
        dict(type='internal_lstm', size=32),
    ]
    return network_spec

def create_baseline_spec():
    baseline_spec = [
        {
            "type": "lstm",
            "size": 32,
        },
        dict(type='dense', size=32, activation='relu'),
        dict(type='dense', size=32, activation='relu'),
    ]
    return baseline_spec
```

### Running 
[Verbose] While training or testing, 
* environment will print out (current_tick , # Long, # Short, Portfolio)
  
[Portfolio]  
* initial portfolio starts with 100*10000(krw-won)     
* reflects change in portfolio value if the agent had invested 100% of its balance every time it opened a position.       
  
[Reward] 
* simply pct earning per trade.    

### Inital Result

#### Portfolio Value Change, Max DrawDown period in Red
![trade](https://github.com/miroblog/tf_deep_rl_trader/blob/master/portfolio_change.png)  

* portfolio value 1000000 -> 1586872.1775 in 56 days

Not bad but the agent definitely needs more
* training data and 
* degree of freedom  (larger network)
  
Beaware of overfitting ! 

## Authors

* **Lee Hankyol** - *Initial work* - [tf_deep_rl_trader](https://github.com/miroblog/tf_deep_rl_trader)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
