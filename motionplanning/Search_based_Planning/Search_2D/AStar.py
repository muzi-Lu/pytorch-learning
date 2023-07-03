import os
from re import S
import sys
import math
import math
import heapq # 堆算法

sys.path.append(os.path.abspath(__file__) + "/../../Search_based_planning")
# 模块搜索路径是一个列表，Python在导入模块时会在其中查找。
# 默认情况下，它包括当前目录和在PYTHONPATH环境变量中指定的目录。

from Search_2D import plotting, env


"""
A-Star progress:

1. 将起点A添加到open列表中
2. 检查open列表，选取花费F最小的节点M(检查M如果为终点则结束寻路，如果open列表没有则寻路失败)
3. 对于与M相邻的每一个节点N
    (1)如果是closed列表，不管
    (2)如果N在closed列表中，不管
    (3)如果N不在open列表中，添加它然后计算出它的花费F(n) = G + H
    (4)如果N在open列表中，当我们使用当前生成路径的时候，检查F花费是否更小，如果是，更新它的花费和它的父节点
4. 重复2, 3步
5. 停止， 当你把终点加入到了openlist中，此时的路径已经找到了，或者查找重点失败，并且openlist是空的，此时没有路径
6. 保存路径
"""

class AStar:
    def __init__(self, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        
        self.Env = env.Env()

        self.u_set = self.Env.motions
        self.obs = self.Env.obs

        # 将路径规划过程中待检测的节点存放于Open List中
        self.OPEN = [] # priority queue / OPEN set
        # 将路径规划中已经检查过的节点放到Close List中
        self.CLOSED = [] # CLOSED set / VISITED order
        # 在路径规划中用于回溯的节点
        self.PARENT = dict() # recorded parent
        self.g = dict() # cost to come


    def searching(self):
        '''
        A_Star Searching.
        :return: path, visited order
        '''

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                        (self.f_value(self.s_start), self.s_start)) # 这个堆是干什么的，需要复习一波了

        while self.OPEN:
            break


def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    astar = AStar(s_start, s_goal, "edclidean")
    plot = plotting.Plotting(s_start, s_goal)

    path, visited = astar.searching()

if __name__ == '__main__':
    main()