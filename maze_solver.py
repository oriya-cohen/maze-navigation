import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Optional


class Maze:
    def __init__(self, shape: tuple = (100, 100), path_prob: float = 0.6):
        self.shape = np.array(shape)
        self.wall = 1
        self.not_wall = 0
        self.path_prob = path_prob
        self.init_pos = np.array([2, 2])
        self.end_pos = self.shape - 3
        self.maze = self._generate()
        self.solvers: List['Solver'] = []

    def _generate(self) -> np.ndarray:
        np.random.seed(1984)
        maze = np.ones(self.shape) * self.not_wall
        maze[np.random.rand(*self.shape) > self.path_prob] = 1
        maze[:, [0, -1]] = self.wall
        maze[[0, -1], :] = self.wall
        maze[tuple(self.init_pos)] = self.not_wall
        maze[tuple(self.end_pos)] = self.not_wall
        return maze

    def add_solver(self, solver: 'Solver') -> None:
        self.solvers.append(solver)
        solver.position = self.init_pos.copy()
        solver.maze_update(self.maze)

    def update_players(self) -> None:
        for solver in self.solvers:
            solver.update_position(self.maze)

    def save_as_image(self, path: list, output_dir: str) -> None:
        plt.imshow(np.array(self.maze)*(-1), cmap='gray')
        plt.plot(path[0][1], path[0][0], 'ro')
        current_pos = path[0]
        for pos in path:
            plt.plot(pos[1],  pos[0],'ro')
            plt.plot([current_pos[1], pos[1]], [current_pos[0], pos[0]], 'r-')
            current_pos = pos
        plt.plot(path[-1][1], path[-1][0], 'go')
        plt.axis('off')
        filename = f"{output_dir}/maze_path.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


class Solver:
    def __init__(self, solver='random'):
        self.position: Optional[np.ndarray] = None

    def maze_update(self, maze: np.ndarray) -> None:
        pass  # Placeholder for internal state logic

    def update_position(self, maze: np.ndarray) -> None:
        pass  # Placeholder for movement logic


class AStarSolver:
    def __init__(self, heuristic: Callable[[np.ndarray, np.ndarray], float]):
        self.heuristic = heuristic
        b = np.sqrt(2)
        self.dist = np.array([[b, 1, b],
                              [1, 0, 1],
                              [b, 1, b]])
        self.open = []
        self.closed = []

    def solve(self, maze: Maze, start: np.ndarray, goal: np.ndarray):
        self.start, self.goal = start, goal
        def h(pos): return self.heuristic(pos, goal)
        g_start = 0
        parent_init = None
        self.open.append((g_start, h(start), start, parent_init))
        neighbors = [[-1,+1],   # 1
                     [-1, 0],   # 2
                     [-1,-1],   # 3     neighbors
                     [+1,+1],   # 4    | 1  7  4 |
                     [+1, 0],   # 5    | 2  X  5 |
                     [+1,-1],   # 6    | 3  8  6 |
                     [ 0,+1],   # 7
                     [ 0,-1]]   # 8
        # TODO: Implement full A* logic
        while self.open:
            current = self.open.pop()   # Pick node current from open list with the lowest f(n)
            print('checking ' + str(current[2]) + " with f=" + str(current[0]+current[1]))
            # move current to closed
            self.closed.append(current)
            if all(current[2] == maze.end_pos):  # make sure comparison is valide between np arrays
                return self.return_path()


            for neighbor in neighbors:
                pos = current[2] + neighbor
                if maze.maze[*pos] == maze.wall:
                    continue
                if any([not(any(pos - closed_check[2]))   # pos in closed list
                        for closed_check in self.closed]):
                    continue
                center = np.array((1, 1))
                tentative_g = self.dist[*(center + neighbor)] + current[0]
                new_node = (tentative_g, h(pos), pos, current[2])
                pos_opened = [list(node[2]) for node in self.open]

                # If neighbor not in open list or tentative_g < g(neighbor):
                if list(new_node[2]) in pos_opened:
                    new_node_pos_in_list = pos_opened.index(list(new_node[2]))
                    if self.open[new_node_pos_in_list][0] > new_node[0]:
                        self.open[new_node_pos_in_list] = new_node  # update node
                else:
                    self.open.append(new_node)
            self.open.sort(key=lambda x:x[0]+x[1], reverse=True)

        raise('no solution for maze: ' + str(maze.maze))

        pass

    def return_path(self):
        origin_dict = {str(node[2].tolist()):node[3].tolist() if type(node[3]) is np.ndarray
                        else None for node in self.closed}
        start_pos = self.start.tolist()
        current_pos = self.goal.tolist()
        path = []
        while str(current_pos) not in str(start_pos):
            path.append(current_pos)
            current_pos = origin_dict[str(current_pos)]
        path.append(current_pos)
        path.reverse()
        return path


def manhattan(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.sum(np.abs(p1 - p2)))

def auclidean_dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.sum((p1 - p2)**2)**.5)

def main():
    maze = Maze(shape=(30, 30))

    solver = AStarSolver(heuristic=auclidean_dist)
    path = solver.solve(maze, maze.init_pos, maze.end_pos)

    maze.save_as_image(path, output_dir="./outputs")


if __name__ == "__main__":
    main()
