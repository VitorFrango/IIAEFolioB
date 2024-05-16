import heapq
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import pandas as pd

# Custos de deslocação
custo_dist = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4, 5: 8, 6: 10}
max_dist = 6

# Definição da matriz de zonas
matriz = [
    # ID1  5x5
    [
        [0, 7, 0, 0, 4],
        [0, 0, 0, 4, 0],
        [1, 0, 0, 0, 0],
        [4, 4, 1, 0, 0],
        [6, 0, 3, 4, 4],
    ],
    # ID2  5x5
    [
        [4, 0, 0, 10, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 6, 3],
        [0, 4, 0, 0, 2],
        [8, 0, 6, 3, 0],
    ],
    # ID3 7x7
    [
        [0, 8, 0, 4, 5, 10, 0],
        [0, 4, 0, 7, 0, 4, 0],
        [0, 2, 4, 2, 0, 0, 2],
        [0, 7, 0, 1, 2, 0, 0],
        [2, 4, 0, 0, 3, 0, 2],
        [0, 4, 0, 0, 3, 0, 0],
        [2, 0, 0, 0, 0, 0, 0],
    ],
    # ID4 7x7
    [
        [0, 0, 1, 0, 7, 0, 1],
        [0, 1, 4, 0, 0, 0, 4],
        [0, 0, 0, 0, 2, 0, 0],
        [3, 1, 0, 8, 5, 7, 7],
        [0, 4, 0, 3, 0, 0, 0],
        [0, 0, 0, 3, 2, 4, 2],
        [0, 8, 3, 6, 3, 0, 0],
    ],
    # ID5 9x9
    [
        [6, 7, 2, 0, 0, 0, 0, 0, 0],
        [3, 3, 6, 0, 8, 4, 3, 1, 0],
        [0, 0, 8, 0, 0, 0, 2, 4, 0],
        [0, 0, 0, 1, 0, 3, 2, 0, 0],
        [0, 0, 0, 7, 4, 0, 1, 0, 0],
        [12, 8, 0, 5, 4, 1, 4, 3, 4],
        [8, 0, 1, 2, 4, 3, 3, 0, 0],
        [1, 1, 0, 0, 0, 0, 5, 0, 0],
        [4, 0, 0, 0, 4, 6, 0, 13, 2],
    ],
    # ID6 9x9
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 8, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 1, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 3, 0],
        [0, 0, 2, 4, 0, 0, 0, 1, 0],
        [0, 2, 0, 0, 8, 0, 4, 3, 10],
        [0, 0, 3, 0, 0, 4, 0, 0, 0],
    ],
    # ID7 11x11
    [
        [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
        [0, 0, 11, 2, 0, 0, 9, 3, 0, 0, 3],
        [0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0],
        [4, 1, 2, 3, 0, 4, 0, 0, 4, 0, 0],
        [5, 0, 0, 0, 4, 0, 1, 0, 4, 3, 0],
        [0, 0, 0, 7, 4, 0, 1, 0, 0, 7, 0],
        [0, 8, 0, 0, 0, 0, 3, 0, 1, 0, 3],
        [0, 3, 0, 0, 5, 2, 3, 0, 0, 0, 2],
        [0, 0, 0, 3, 1, 0, 2, 8, 0, 0, 0],
        [0, 3, 4, 0, 7, 0, 0, 7, 0, 0, 0],
        [4, 2, 0, 4, 0, 3, 0, 0, 5, 7, 0],
    ],
    # ID8 11x11
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 10, 10, 0, 0, 0, 4, 5, 0, 0],
        [0, 4, 1, 0, 8, 0, 0, 0, 0, 0, 5],
        [8, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 13, 0, 0, 0, 2, 0, 3],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0],
        [4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    # ID9 13x13
    [
        [2, 4, 0, 0, 6, 7, 3, 4, 0, 0, 3, 0, 1],
        [0, 0, 2, 0, 3, 0, 0, 6, 0, 0, 8, 11, 3],
        [0, 3, 0, 8, 0, 0, 2, 0, 0, 0, 0, 0, 4],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0],
        [0, 6, 0, 8, 0, 3, 0, 0, 0, 0, 0, 0, 1],
        [0, 3, 0, 2, 0, 0, 9, 0, 0, 0, 0, 5, 6],
        [1, 9, 4, 0, 0, 2, 4, 0, 0, 0, 3, 2, 0],
        [2, 3, 0, 4, 0, 0, 0, 6, 2, 0, 1, 0, 3],
        [0, 0, 0, 0, 0, 6, 0, 0, 0, 2, 2, 0, 8],
        [7, 2, 4, 2, 0, 0, 6, 4, 1, 0, 0, 0, 7],
        [0, 0, 0, 11, 0, 0, 0, 0, 3, 4, 0, 9, 0],
        [0, 0, 0, 0, 1, 4, 3, 4, 0, 0, 0, 3, 11],
        [0, 0, 4, 7, 7, 0, 0, 2, 0, 2, 5, 0, 1],
    ],
    # ID10 13x13
    [
        [0, 0, 1, 4, 0, 0, 9, 0, 0, 0, 12, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 9, 4, 0, 0, 0, 6, 0, 0],
        [0, 6, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 6, 10, 0, 1, 4],
        [0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 1, 3, 0, 0, 0, 0, 9, 0, 0, 0],
        [9, 0, 0, 3, 3, 0, 0, 0, 0, 3, 4, 0, 0],
        [0, 1, 4, 0, 0, 0, 0, 0, 0, 5, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 10],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
    ],
    # ID11 15x13
    [
        [0, 0, 0, 4, 0, 0, 0, 6, 0, 0, 0, 0, 2, 2, 0],
        [0, 2, 12, 0, 3, 0, 0, 0, 0, 26, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 3, 0, 0, 6, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 5, 4, 0, 0, 3, 0, 0, 0],
        [9, 12, 0, 0, 0, 4, 1, 6, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 2, 0, 0, 0, 7, 0, 4, 0, 0, 0],
        [0, 0, 2, 0, 0, 9, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 2, 16, 0, 8, 0, 2, 0, 0, 0, 0, 7],
        [0, 0, 5, 0, 6, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0],
    ],
    # ID12 15x13
    [
        [0, 0, 0, 0, 0, 0, 0, 10, 3, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 10, 3, 8, 11, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
        [0, 4, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 5, 0, 10, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 8, 0, 15],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1, 0, 2, 0],
        [0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        [8, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 2, 0, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1],
    ],
    # ID13 17x13
    [
        [0, 0, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 4, 2, 0, 3, 0, 0, 0, 0, 0],
        [6, 0, 3, 0, 0, 0, 6, 0, 30, 0, 1, 8, 6, 10, 0, 0, 0],
        [0, 7, 0, 1, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 4],
        [0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 8, 0, 0, 0, 0, 0, 3, 0, 0, 36, 0, 1, 0, 0, 2, 0],
        [6, 0, 0, 0, 8, 2, 8, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [8, 1, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 6, 7, 0, 0],
        [3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 4, 0, 0, 1],
        [3, 0, 0, 2, 0, 4, 0, 0, 0, 0, 9, 0, 0, 0, 8, 16, 24],
        [0, 1, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 6, 1, 0],
        [0, 3, 4, 0, 3, 4, 0, 10, 0, 0, 0, 0, 5, 5, 8, 4, 4],
        [8, 0, 0, 0, 0, 0, 17, 0, 0, 10, 0, 2, 0, 0, 2, 0, 0],
    ],
    # ID14 17x13
    [
        [0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0],
        [0, 6, 0, 0, 0, 0, 8, 0, 10, 0, 0, 0, 0, 2, 2, 3, 0],
        [0, 0, 0, 0, 0, 4, 0, 8, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 8, 1, 0, 0, 7, 0, 0, 0, 0, 0, 5, 3, 0, 0, 0],
        [0, 0, 3, 0, 1, 0, 0, 3, 0, 0, 3, 0, 3, 0, 8, 0, 0],
        [0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 6, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 1, 3, 0, 1, 0, 4, 0, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [4, 8, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 6, 3],
    ],
    # ID15 19x13
    [
        [0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 8, 0, 6, 0, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 2, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1],
        [2, 0, 8, 3, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 2, 1, 4, 0],
        [0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 18, 10, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 7, 4, 0, 0, 4, 3],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 2],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 2, 2, 0, 0, 4, 0, 0, 10, 1, 0],
        [3, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 4, 0, 0, 1, 0],
        [2, 0, 2, 0, 0, 0, 0, 1, 0, 0, 4, 1, 0, 3, 0, 0, 0, 3, 3],
        [0, 0, 0, 0, 4, 0, 1, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 4, 2, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, 2, 0, 3, 22, 0, 0, 0, 0, 0, 2, 7, 0, 0, 0, 0, 0, 1],
        [0, 9, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 4, 0, 8],
    ],
    # ID16 19x13
    [
        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [5, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 2, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 2],
        [0, 0, 3, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [5, 0, 0, 0, 0, 0, 6, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 4, 0, 0, 5, 0, 0, 0, 0, 0, 4, 1],
        [3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 7, 2, 0, 0, 0, 1, 0, 3],
        [0, 0, 1, 0, 0, 4, 11, 0, 3, 0, 0, 0, 0, 11, 3, 0, 0, 0, 0],
        [1, 0, 2, 8, 0, 0, 0, 0, 0, 0, 4, 0, 0, 3, 1, 0, 0, 0, 0],
        [3, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    ],
    # ID17 19x15
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 16, 1, 0, 5, 0, 3, 0, 0, 0, 4, 0],
        [0, 2, 3, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0],
        [0, 9, 1, 0, 0, 0, 0, 4, 2, 2, 0, 1, 8, 2, 0, 4, 24, 10, 13],
        [0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 4, 0, 11, 0, 0, 0, 2, 1, 1],
        [0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 2],
        [0, 3, 12, 0, 4, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 9, 0, 0, 0, 0, 0, 8, 4, 0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 2, 0, 0, 6, 0, 3, 0, 6, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 12, 2, 4, 0, 0, 7, 0, 1],
        [6, 4, 4, 0, 0, 8, 0, 3, 2, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 8, 0, 0, 2, 0, 0, 8, 0, 0, 0, 0, 2],
        [0, 4, 18, 0, 0, 0, 0, 0, 0, 0, 4, 1, 2, 0, 0, 0, 0, 8, 3],
        [2, 0, 7, 0, 7, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 7, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 11, 0, 30, 0],
        [1, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
    ],
    # ID18 19x15
    [
        [0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 7, 0, 0, 0, 0, 0],
        [0, 2, 5, 7, 2, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1],
        [0, 7, 0, 2, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 4, 2, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 5, 0, 4, 0, 0, 3, 4, 0, 0, 0, 3, 0, 0, 0, 0, 7, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 6, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0],
        [0, 0, 7, 0, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 8, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0, 8, 0, 3],
        [8, 0, 0, 0, 0, 0, 2, 6, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 4, 0, 0, 8, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 0, 13, 0, 2, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0],
        [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 3],
    ],
    # ID19 19x17
    [
        [0, 2, 0, 0, 0, 4, 4, 0, 0, 4, 0, 0, 1, 6, 0, 1, 4, 0, 0],
        [1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 9, 3, 0, 0, 0, 0, 0],
        [3, 0, 0, 4, 0, 9, 1, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 3, 0, 6, 0, 0, 0, 0, 3, 0, 0, 11, 17, 0, 0, 0, 0],
        [6, 0, 1, 0, 0, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 6, 0, 0, 1, 0, 0, 0, 2, 0],
        [1, 0, 10, 0, 0, 2, 2, 0, 3, 4, 8, 0, 0, 9, 11, 1, 0, 16, 0],
        [3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0],
        [0, 6, 0, 1, 0, 0, 0, 0, 3, 5, 0, 0, 2, 4, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 6, 0, 3, 6, 0, 10, 6, 0, 0, 0, 0, 0, 0, 2],
        [3, 0, 0, 4, 4, 0, 2, 0, 0, 0, 1, 0, 0, 1, 2, 16, 11, 0, 0],
        [7, 0, 0, 3, 0, 0, 0, 0, 0, 10, 12, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 6, 3, 0, 0, 0, 0, 0, 7, 0, 0],
        [0, 0, 1, 0, 4, 8, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 7, 0, 0, 0, 0, 2],
    ],
    # ID20 19x17
    [
        [3, 4, 0, 0, 3, 0, 0, 0, 0, 6, 0, 4, 4, 0, 0, 0, 4, 0, 0],
        [4, 0, 0, 5, 0, 0, 0, 0, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 9, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
        [4, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 3, 0],
        [2, 0, 7, 0, 0, 11, 0, 0, 0, 0, 0, 0, 5, 0, 7, 0, 0, 0, 0],
        [9, 0, 0, 0, 1, 0, 1, 15, 0, 0, 0, 0, 1, 0, 0, 0, 1, 4, 3],
        [0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 7, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0],
        [0, 0, 6, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 3],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 7, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ],
]


# Função para calcular o custo de deslocação
def calcular_custo_deslocacao(estacoes, matriz):
    n, m = len(matriz), len(matriz[0])
    distancias = np.full((n, m), np.inf)

    for (ex, ey) in estacoes:
        for i in range(n):
            for j in range(m):
                dist = max(abs(ex - i), abs(ey - j))
                distancias[i][j] = min(distancias[i][j], dist)

    custo_total = 0
    num_familias = 0
    for i in range(n):
        for j in range(m):
            if matriz[i][j] > 0:
                custo_total += matriz[i][j] * custo_dist[min(distancias[i][j], max_dist)]
                num_familias += matriz[i][j]

    custo_medio = custo_total / num_familias if num_familias > 0 else 0
    return custo_medio


# Função heurística
def heuristica(estacoes, matriz):
    custo_medio = calcular_custo_deslocacao(estacoes, matriz)
    return len(estacoes) * 1000 + 100 * custo_medio


# Função para encontrar a melhor posição para uma nova estação
def melhor_posicao_para_nova_estacao(estacoes, matriz):
    n, m = len(matriz), len(matriz[0])
    melhor_custo = float('inf')
    melhor_posicao = None

    for i in range(n):
        for j in range(m):
            nova_estacao = (i, j)
            if nova_estacao not in estacoes:
                novas_estacoes = estacoes + [nova_estacao]
                novo_custo = heuristica(novas_estacoes, matriz)
                if novo_custo < melhor_custo:
                    melhor_custo = novo_custo
                    melhor_posicao = nova_estacao

    return melhor_posicao, melhor_custo


# Algoritmo A* com abordagem melhorativa
def a_star_melhorativo(matriz, max_time=60000, max_evaluations=100000):
    estacoes = []
    custo_inicial = heuristica(estacoes, matriz)
    fronteira = [(custo_inicial, estacoes)]
    visitados = set()

    num_visualizacoes = 0
    num_geracoes = 0
    start_time = time.time()

    while fronteira:
        custo_atual, estacoes = heapq.heappop(fronteira)
        num_visualizacoes += 1

        if (tuple(estacoes), custo_atual) in visitados:
            continue

        visitados.add((tuple(estacoes), custo_atual))

        # Verificar se a solução é válida
        custo_medio = calcular_custo_deslocacao(estacoes, matriz)
        if custo_medio < 3:
            end_time = time.time()
            tempo_execucao = (end_time - start_time) * 1000
            return estacoes, len(estacoes), custo_medio, num_visualizacoes, num_geracoes, tempo_execucao

        # Encontrar a melhor posição para adicionar uma nova estação
        melhor_posicao, melhor_custo = melhor_posicao_para_nova_estacao(estacoes, matriz)
        if melhor_posicao:
            novas_estacoes = estacoes + [melhor_posicao]
            heapq.heappush(fronteira, (melhor_custo, novas_estacoes))
            num_geracoes += 1

        # Critérios de paragem
        if num_geracoes >= max_evaluations or (time.time() - start_time) * 1000 >= max_time:
            end_time = time.time()
            tempo_execucao = (end_time - start_time) * 1000
            return estacoes, len(estacoes), custo_medio, num_visualizacoes, num_geracoes, tempo_execucao

    end_time = time.time()
    tempo_execucao = (end_time - start_time) * 1000
    return None, num_visualizacoes, num_geracoes, tempo_execucao


# Tabela de resultados
resultados = {
    "Instância": [],
    "Avaliações": [],
    "Gerações": [],
    "Custo": [],
    "Tempo (msec)": [],
    "Melhor resultado": []
}

# Executar o algoritmo para todas as matrizes
for idx, matriz_id in enumerate(matriz):
    resultado = a_star_melhorativo(matriz_id)
    if resultado and resultado[0] is not None:
        print(f"-------------------------------------------")
        print(f"Instancia ID {idx + 1}:")
        print(f"Melhor localização das estações: {resultado[0]}")
        print(f"Número de estações (A): {math.ceil(resultado[1])}")
        print(f"Custo médio de deslocação (B): {math.ceil(resultado[2])}")
        print(f"Número de visualizações: {math.ceil(resultado[3])}")
        print(f"Número de gerações: {math.ceil(resultado[4])}")
        print(f"Tempo de execução: {math.ceil(resultado[5])} msec")
        print(f"Custo da solução: {math.ceil(resultado[1] * 1000 + resultado[2] * 100)}")
        print(f"-------------------------------------------")

        # Add results to the table
        resultados["Instância"].append(idx + 1)
        resultados["Avaliações"].append(resultado[3])
        resultados["Gerações"].append(resultado[4])
        resultados["Custo"].append(math.ceil(resultado[2]))
        resultados["Tempo (msec)"].append(math.ceil(resultado[5]))
        resultados["Melhor resultado"].append(math.ceil(resultado[1] * 1000 + resultado[2] * 100))
    else:
        print(f"Instancia ID {idx + 1}: Nenhuma solução encontrada")
        print(f"Número de visualizações: {math.ceil(resultado[1])}")
        print(f"Número de gerações: {math.ceil(resultado[2])}")
        print(f"Tempo de execução: {math.ceil(resultado[3])} msec")

        # Add results to the table
        resultados["Instância"].append(idx + 1)
        resultados["Avaliações"].append(resultado[1])
        resultados["Gerações"].append(resultado[2])
        resultados["Custo"].append("N/A")
        resultados["Tempo (msec)"].append(resultado[3])
        resultados["Melhor resultado"].append("N/A")

# Create a DataFrame from the results
df = pd.DataFrame(resultados)
df = df.set_index("Instância")

# Plotting the table
fig, ax = plt.subplots(figsize=(15, 8))
ax.axis('tight')
ax.axis('off')

# Constructing a multi-level header
header = pd.MultiIndex.from_product([['Algoritmo 1 / configurações 1'], df.columns.tolist()])
df.columns = header

# Create the table
the_table = ax.table(cellText=df.values,
                     colLabels=df.columns.levels[1],
                     rowLabels=df.index,
                     cellLoc='center',
                     loc='center')

# Set the column headers
header_labels = ['Instância', 'Avaliações', 'Gerações', 'Custo', 'Tempo (msec)', 'Melhor resultado']
num_columns = len(df.columns)  # Get the number of columns in the DataFrame

for i in range(min(num_columns, len(header_labels))):  # Ensure i does not exceed the number of columns
    cell = the_table.get_celld()[(0, i)]
    cell.get_text().set_text(header_labels[i])
    cell.get_text().set_fontsize(12)
    cell.get_text().set_fontweight('bold')
    cell.get_text().set_ha('center')
    cell.get_text().set_va('center')

# Adjusting cell size and font size
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
for key, cell in the_table.get_celld().items():
    cell.set_height(0.065)
    cell.set_width(0.2)

plt.show()
