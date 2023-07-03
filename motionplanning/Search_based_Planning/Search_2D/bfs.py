"""
Breadth-first Searching_2D (BFS)
(BFS) is Breadth-first Searching not
"""

import os
import sys
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import plotting, env
from Search_2D.AStar import AStar
import math
import heapq # 堆算法优化

