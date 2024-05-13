import heapq


class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.x = 0
        self.y = 0


class Graph:
    def __init__(self, size):
        self.adj = {}
        self.nodes = [Node(i) for i in range(size)]

    def add_edge(self, from_node, to_node, cost):
        if cost != 0:
            if from_node.id not in self.adj:
                self.adj[from_node.id] = []
            self.adj[from_node.id].append((to_node, cost))

    def load_from_matrix(self, matrix):
        size = len(matrix)
        for i in range(size):
            for j in range(size):
                self.add_edge(self.nodes[i], self.nodes[j], matrix[i][j])

    def get_neighbors(self, node):
        return self.adj.get(node.id, [])


class AStarSearch:
    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.open_set = []
        heapq.heappush(self.open_set, (0, start))
        self.g_score = {start.id: 0}
        self.came_from = {}

    def heuristic(self, a, b):
        return abs(a.id - b.id)

    def reconstruct_path(self, current):
        path = []
        while current is not None:
            path.append(current)
            current = self.came_from.get(current.id)
        path.reverse()
        return path

    def search(self):
        while self.open_set:
            current = heapq.heappop(self.open_set)[1]

            if current == self.goal:
                path = self.reconstruct_path(current)
                print(" -> ".join(str(node.id) for node in path) + " chegou ao destino")
                return

            for neighbor, cost in self.graph.get_neighbors(current):
                tentative_g_score = self.g_score[current.id] + cost

                if tentative_g_score < self.g_score.get(neighbor.id, float('inf')):
                    self.came_from[neighbor.id] = current
                    self.g_score[neighbor.id] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, self.goal)
                    heapq.heappush(self.open_set, (f_score, neighbor))


# Exemplo de uso
matrix5x5 = [
    [0, 7, 0, 0, 4],
    [0, 0, 0, 4, 0],
    [1, 0, 0, 0, 0],
    [4, 4, 1, 0, 0],
    [6, 0, 3, 4, 4]
]

graph5x5 = Graph(5)
graph5x5.load_from_matrix(matrix5x5)
start5x5 = graph5x5.nodes[0]
goal5x5 = graph5x5.nodes[4]
aStar5x5 = AStarSearch(graph5x5, start5x5, goal5x5)
aStar5x5.search()
