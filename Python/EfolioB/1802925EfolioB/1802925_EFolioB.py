
"""
    EfolioB - Introdução à Inteligência Artificial
    Implementação do algoritmo A* nelhorativo para resolução de problemas de caminho mínimo
    Vitor Manuel Frango
    1802925
"""

import heapq
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from termcolor import colored


# Definição dos custos de deslocação
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

def calcular_custo_deslocacao(estacoes, matriz):
    """
    Calcula o custo médio de deslocação
    Args:
        estacoes (list of tuple): Lista de coordenadas das estações.
        matriz (list of list): Matriz de zonas com o número de famílias em cada posição.
    Returns:
        float: O custo médio de deslocação.
    """
    n, m = len(matriz), len(matriz[0])
    distancias = np.full((n, m), np.inf)

    for (ex, ey) in estacoes:
        for i in range(n):
            for j in range(m):
                dist = max(abs(ex - i), abs(ey - j))
                distancias[i][j] = min(distancias[i][j], dist) # Distância mínima a uma estação

    custo_total = 0
    num_familias = 0
    for i in range(n):
        for j in range(m):
            if matriz[i][j] > 0:
                # Custo de deslocação
                custo_total += matriz[i][j] * custo_dist[min(distancias[i][j], max_dist)]
                num_familias += matriz[i][j]

    return custo_total / num_familias if num_familias > 0 else 0

def heuristica(estacoes, matriz, peso_estacao=1000, peso_custo=100):
    """
    Calcula uma estimativa do custo de uma solução (combinação de estações).
    Args:
        estacoes (list of tuple): Lista de coordenadas das estações.
        matriz (list of list): Matriz de zonas com o número de famílias em cada posição.
        peso_estacao (int): Peso atribuído ao número de estações na heurística.
        peso_custo (int): Peso atribuído ao custo médio de deslocação na heurística.
    Returns:
        float: A estimativa do custo da solução.
    """
    custo_medio = calcular_custo_deslocacao(estacoes, matriz)
    return len(estacoes) * peso_estacao + peso_custo * custo_medio

def melhor_posicao_para_nova_estacao(estacoes, matriz):
    """
    Encontra a melhor posição para adicionar uma nova estação, minimizando o custo.
    Args:
        estacoes (list of tuple): Lista de coordenadas das estações.
        matriz (list of list): Matriz de zonas com o número de famílias em cada posição.
    Returns:
        tuple: A melhor posição (x, y) e o custo resultante.
    """
    n, m = len(matriz), len(matriz[0])
    melhor_custo = float('inf')
    melhor_posicao = None


    # Verificar todas as posições da matriz
    for i in range(n):
        for j in range(m):
            if (i, j) not in estacoes:
                novas_estacoes = estacoes + [(i, j)]
                novo_custo = heuristica(novas_estacoes, matriz)
                if novo_custo < melhor_custo:
                    melhor_custo = novo_custo
                    melhor_posicao = (i, j)

    return melhor_posicao, melhor_custo

def a_star_melhorativo(matriz, max_time=60000, max_evaluations=100000):
    """
    Executa o algoritmo A* melhorativo para encontrar a melhor combinação de estações.
    Args:
        matriz (list of list): Matriz de zonas com o número de famílias em cada posição.
        max_time (int): Tempo máximo de execução em milissegundos.
        max_evaluations (int): Número máximo de avaliações de soluções.
    Returns:
        tuple: A melhor solução, número de avaliações, número de nós gerados, tempo de execução, custo médio e custo da solução.
    """
    estacoes = []
    custo_inicial = heuristica(estacoes, matriz)
    fronteira = [(custo_inicial, estacoes)]
    visitados = set()
    melhor_solucao = None
    melhor_custo_solucao = float('inf')

    num_avaliacoes = 0
    num_nos_gerados = 0
    start_time = time.perf_counter()

    while fronteira and num_avaliacoes < max_evaluations and (time.perf_counter() - start_time) * 1000 < max_time:
        custo_atual, estacoes = heapq.heappop(fronteira)
        num_avaliacoes += 1

        if (tuple(estacoes), custo_atual) in visitados:
            continue
        visitados.add((tuple(estacoes), custo_atual))

        custo_medio = calcular_custo_deslocacao(estacoes, matriz)
        if custo_medio < 3 and custo_atual < melhor_custo_solucao:
            melhor_solucao = estacoes
            melhor_custo_solucao = custo_atual

        melhor_posicao, melhor_custo = melhor_posicao_para_nova_estacao(estacoes, matriz)
        if melhor_posicao:
            novas_estacoes = estacoes + [melhor_posicao]
            heapq.heappush(fronteira, (melhor_custo, novas_estacoes))
            num_nos_gerados += 1

    tempo_execucao = (time.perf_counter() - start_time) * 1000
    # Verificar se a solução é válida e foi encontrada em menos de 1 minuto
    if melhor_solucao and tempo_execucao <= 60000:
        custo_medio = calcular_custo_deslocacao(melhor_solucao, matriz)
        custo_da_solucao = len(melhor_solucao) * 1000 + 100 * custo_medio
        return melhor_solucao, num_avaliacoes, num_nos_gerados, tempo_execucao, custo_medio, custo_da_solucao

    return None, num_avaliacoes, num_nos_gerados, tempo_execucao, None, None

# Tabela de resultados
resultados = {
    "Instância": [],
    "Avaliações": [],
    "Gerações": [],
    "Custo": [],
    "Tempo (msec)": [],
    "Melhor resultado": [],
}

# Dicionário para armazenar as melhores soluções
melhores_solucoes = {}

# Executar o algoritmo para todas as matrizes
for idx, matriz_id in enumerate(matriz):
    resultado = a_star_melhorativo(matriz_id)
    if resultado and resultado[0] is not None:
        # Armazenar a melhor solução
        if idx + 1 not in melhores_solucoes or resultado[5] < melhores_solucoes[idx + 1][5]:
            melhores_solucoes[idx + 1] = resultado

        print(f"-------------------------------------------")
        print(f"Instancia ID {idx + 1}:")

        # Apresentar a solução no formato desejado
        n, m = len(matriz_id), len(matriz_id[0])
        solucao_formatada = np.array(matriz_id).astype(str)
        for x, y in resultado[0]:
            solucao_formatada[x][y] += "#"
        for linha in solucao_formatada:
            linha_colorida = []
            for celula in linha:
                if "#" in celula:
                    celula = colored(celula, 'red')
                linha_colorida.append(celula)
            print(" ".join(linha_colorida))

        # Imprimir métricas da solução
        print(f"-------------------------------------------")
        print(f"Avaliações: {resultado[1]}")
        print(f"Gerações: {resultado[2]}")
        print(f"Custo: {resultado[5]:.0f}")
        print(f"Tempo: {resultado[3]:.0f} msec")
        print(f"-------------------------------------------")
        print(f"Número de estações (A): {len(resultado[0])}")
        print(f"Custo médio de deslocação (B): {resultado[4]:.3f}")
        print(f"-------------------------------------------")

        # Adicionar resultados à tabela
        resultados["Instância"].append(idx + 1)
        resultados["Avaliações"].append(resultado[1])
        resultados["Gerações"].append(resultado[2])
        resultados["Custo"].append(f"{resultado[4]:.3f}")
        resultados["Tempo (msec)"].append(f"{resultado[3]:.0f}")
        resultados["Melhor resultado"].append(f"{resultado[5]:.0f}")
    else:
        # Apresentar resultados para instâncias sem solução válida
        print(f"Instancia ID {idx + 1}: Nenhuma solução válida encontrada em 1 minuto")
        print(f"Avaliações: {resultado[1]}")
        print(f"Gerações: {resultado[2]}")
        print(f"Tempo de execução: {resultado[3]:.0f} msec")
        print(f"-------------------------------------------")

        # Adicionar resultados à tabela
        resultados["Instância"].append(idx + 1)
        resultados["Avaliações"].append(resultado[1])
        resultados["Gerações"].append(resultado[2])
        resultados["Custo"].append("N/A")
        resultados["Tempo (msec)"].append(f"{resultado[3]:.0f}")
        resultados["Melhor resultado"].append("N/A")

# Apresentar as melhores soluções válidas encontradas em menos de 1 minuto
print("\nMelhores Soluções:")
for instancia, resultado in melhores_solucoes.items():
    if resultado[3] <= 60000:  # Verificar se a solução foi encontrada em menos de 1 minuto
        print(f"\nInstancia ID {instancia}:")
        n, m = len(matriz[instancia - 1]), len(matriz[instancia - 1][0])
        solucao_formatada = np.array(matriz[instancia - 1]).astype(str)
        for x, y in resultado[0]:
            solucao_formatada[x][y] += "#"
        for linha in solucao_formatada:
            linha_colorida = []
            for celula in linha:
                if "#" in celula:
                    celula = colored(celula, 'red')
                linha_colorida.append(celula)
            print(" ".join(linha_colorida))
        print(f"Custo da solução: {resultado[5]:.0f}")

# Criar um DataFrame com os resultados e mostrar a tabela
df = pd.DataFrame(resultados)
df = df.set_index("Instância")

# Plotar a tabela com os resultados
fig, ax = plt.subplots(figsize=(15, 8))
ax.axis('tight')
ax.axis('off')

# Construir o cabeçalho da tabela
header = pd.MultiIndex.from_product([['Algoritmo 1 / configurações 1'], df.columns.tolist()])
df.columns = header

# Criar a tabela
the_table = ax.table(cellText=df.values,
                     colLabels=df.columns.levels[1],
                     rowLabels=df.index,
                     cellLoc='center',
                     loc='center')

# Construir o cabeçalho da tabela
header_labels = ['Instância', 'Avaliações', 'Gerações', 'Custo', 'Tempo (msec)', 'Melhor resultado']
num_columns = len(df.columns)  # Número de colunas no DataFrame

# Adicionar o cabeçalho à tabela com formatação
for i in range(min(num_columns, len(header_labels))):  # Garantir que i não exceda o número de colunas
    cell = the_table.get_celld()[(0, i)]
    cell.get_text().set_text(header_labels[i])
    cell.get_text().set_fontsize(12)
    cell.get_text().set_fontweight('bold')
    cell.get_text().set_ha('center')
    cell.get_text().set_va('center')

# Ajustar o tamanho das células
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
for key, cell in the_table.get_celld().items():
    cell.set_height(0.065)
    cell.set_width(0.2)


#  retirar o comentário da linha abaixo para mostrar a tabela com os resultados
#plt.show()
