from gymnasium.envs.registration import register

register(
    id='AirHockey-v0',
    entry_point='myenv.env:MyRobotEnv'
)