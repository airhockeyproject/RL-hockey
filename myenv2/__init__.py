from gymnasium.envs.registration import register

register(
    id='AirHockey2-v0',
    entry_point='myenv2.env_hockey:AirHockeyEnv'
)
