//
// Created by Vitor Frango on 10/05/2024.
//


#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <cmath>

using namespace std;

struct Node {
    int id;
    float x, y; // Coordenadas do nó, se necessário para a função heurística

    Node(int id, float x, float y) : id(id), x(x), y(y) {}
};

// Heurística - Distância Euclidiana
float heuristic(const Node* a, const Node* b) {
    return sqrt(pow(a->x - b->x, 2) + pow(a->y - b->y, 2));
}

struct Graph {
    map<int, vector<pair<Node*, float>>> adj;

    void addEdge(Node* from, Node* to, float cost) {
        adj[from->id].push_back(make_pair(to, cost));
    }

    vector<pair<Node*, float>>& getNeighbors(Node* node) {
        return adj[node->id];
    }
};

// A* Search Algorithm structure
struct AStarSearch {
    Graph& graph;
    Node* start;
    Node* goal;

    map<int, float> gScore;
    map<int, Node*> cameFrom;
    priority_queue<pair<float, Node*>, vector<pair<float, Node*>>, greater<pair<float, Node*>>> openSet;

    AStarSearch(Graph& graph, Node* start, Node* goal) : graph(graph), start(start), goal(goal) {
        openSet.push(make_pair(0, start));
        gScore[start->id] = 0;
    }

    vector<Node*> reconstructPath(Node* current) {
        vector<Node*> path;
        while (current != nullptr) {
            path.push_back(current);
            current = cameFrom[current->id];
        }
        reverse(path.begin(), path.end());
        return path;
    }

    // função de busca
    void search() {
        while (!openSet.empty()) {
            Node* current = openSet.top().second;
            openSet.pop();

            if (current == goal) {
                vector<Node*> path = reconstructPath(current);
                // Output do caminho
                for (Node* node : path) {
                    cout << node->id << " -> ";
                }
                cout << "chegou ao destino\n";
                return;
            }

            for (auto& neighborPair : graph.getNeighbors(current)) {
                Node* neighbor = neighborPair.first;
                float cost = neighborPair.second;
                float tentative_gScore = gScore[current->id] + cost;

                if (tentative_gScore < gScore[neighbor->id] || gScore.find(neighbor->id) == gScore.end()) {
                    cameFrom[neighbor->id] = current;
                    gScore[neighbor->id] = tentative_gScore;
                    float fScore = tentative_gScore + heuristic(neighbor, goal);
                    openSet.push(make_pair(fScore, neighbor));
                }
            }
        }
    }
};

int main() {
    Node a(1, 0, 0), b(2, 1, 1), c(3, 2, 2);
    Graph graph;
    graph.addEdge(&a, &b, 1.5);
    graph.addEdge(&b, &c, 1.2);
    graph.addEdge(&a, &c, 2.5);

    AStarSearch aStar(graph, &a, &c);
    aStar.search();

    return 0;
}
