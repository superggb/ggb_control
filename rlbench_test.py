from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
import time 
import matplotlib.pyplot as plt

# class Agent(object):

#     def __init__(self, action_size):
#         self.action_size = action_size

#     def act(self, obs):
#         arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
#         gripper = [1.0]  # Always open
#         return np.concatenate([arm, gripper], axis=-1)


def act(graph,model,sess):
    obs_config = ObservationConfig()
    # obs_config.set_all(True)

    # 这几个参数用来关闭摄像头
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.wrist_camera.set_all(False)
    obs_config.left_shoulder_camera.set_all(False)

    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(
        action_mode, obs_config=obs_config, headless=False)
    env.launch()

    task = env.get_task(ReachTarget)

    training_steps = 150
    episode_length = 60
    obs = None
    gripper = [1.0] # 爪子保持打开

    for i in range(training_steps):
        # 初始化一次任务，包括初始Obs数据
        if i % episode_length == 0:
            print('Reset Episode')
            descriptions, obs = task.reset()
            print(descriptions)
        
        # 网络的输入
        vision = obs.front_rgb
        vision = vision[np.newaxis,:]
        state = obs.joint_positions
        state = state[np.newaxis,:]
        action = None
        feed_dict = {
            model.vision:vision,
            model.state:state,
            model.action:action
        }
        run_ops = [model.predict_out]

        with graph.as_default():
            res = sess.run(run_ops,feed_dict=feed_dict)

        # print(res[0][0].shape)
        # print(gripper.shape)
        # input('check shape')
        action = np.concatenate([res[0][0],gripper],axis=-1)
        obs, reward, terminate = task.step(action)
        if reward == 1:
            print("successful try!")

    print('Done')
    env.shutdown()
    # print(a)
    # plt.imshow(a)
    # plt.show()
    # input('check')
