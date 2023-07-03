'''
Env 2D
'''

from re import S


class Env:
    def __init__(self) -> None:
        self.x_range = 51
        self.y_range = 31

        self.motions = [(-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1),
        (1, 0), (1, 1), (1,-1)]

        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        '''
        Initialize obstacles' positions
        :return : map of obstacles
        '''

        x = self.x_range
        y = self.y_range
        obs = set()

        # 上墙 下墙
        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))

        # 左墙 右墙
        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))

        # 障碍
        for i in range(10, 21):
            obs.add((i, 15))
        for i in range(15):
            obs.add((20, i))

        for i in range(15, 30):
            obs.add((30, i))
        for i in range(16):
            obs.add((40, i))

        return obs