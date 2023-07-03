'''
Plot tools 2D
'''

import os 
import sys
from turtle import color
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning")

from Search_2D import env


class Plotting:
    def __init__(self, xI, xG) -> None:
        self.xI, self.xG = xI, xG
        self.env = env.Env()
        self.obs = self.env.obs_map()


    def update_obs(self, obs):
        self.obs = obs

    
    def animation(self, path, visited, name):
        self.plot_grid(name)
        self.plot_visited(visited)
        self.path(path)
        plt.show()

    
    def plot_grid(self, name):
        obs_x = [x[0] for x in self.obs]
        obs_y = [y[0] for y in self.obs]

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")
        plt.plot(obs_x, obs_y, "sk")
        plt.title(name)
        plt.axis("equal")

    def plot_visited(self, visited, cl='gray'):
        if self.xI in visited:
            visited.remove(self.xI)
        
        if self.xG in visited:
            visited.remove(self.xG)

        count = 0

        for x in visited:
            count += 1
            plt.plot(x[0], x[1], color=cl, marker='o')
            # plt.gcf().canvas.mpl_connect
            plt.gcf().canvas.mpl_connect('key_release_event',
                                        lambda event: [exit(0) if event.key == 'escape' else None])

            if count < len(visited) / 3:
                length = 20
            elif count < len(visited) * 2 / 3:
                length = 30
            else:
                length = 40

            if count % length == 0:
                plt.pause(0.001)
        plt.pause(0.01)

    def plot_path(self, path, cl='r', flag=False):
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]

        if not flag:
            plt.plot(path_x, path_y, linewidth='3', color='r')
        else:
            plt.plot(path_x, path_y, linewidth='3', color='r')
        
        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")

        plt.pause(0.01)

    def animation_lrta(self, path, visited, name):
        pass

    def animation_ara_star(self, path, visited, name):
        pass

    def animation_bi_astar():
        pass
