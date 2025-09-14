import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Optional


class Maze:
    def __init__(self, shape: tuple = (100, 100), path_prob: float = 0.8):
        self.shape = np.array(shape)
        self.path_prob = path_prob
        self.init_pos = np.array([2, 2])
        self.maze = self._generate()
        self.players: List['Player'] = []

    def _generate(self) -> np.ndarray:
        np.random.seed(1984)
        maze = np.zeros(self.shape)
        maze[np.random.rand(*self.shape) > self.path_prob] = 1
        maze[:, [0, -1]] = 1
        maze[[0, -1], :] = 1
        maze[tuple(self.init_pos)] = 2
        maze[tuple(self.shape - 3)] = 3
        return maze

    def add_player(self, player: 'Player') -> None:
        self.players.append(player)
        player.position = self.init_pos.copy()
        player.maze_update(self.maze)

    def update_players(self) -> None:
        for player in self.players:
            player.update_position(self.maze)

    def save_as_image(self, step: int, output_dir: str) -> None:
        plt.imshow(self.maze, cmap='gray')
        for player in self.players:
            plt.plot(player.position[1], player.position[0], 'ro')
        plt.axis('off')
        filename = f"{output_dir}/maze_step_{step:03d}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


class Player:
    def __init__(self):
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

    def solve(self, maze: np.ndarray, start: np.ndarray, goal: np.ndarray):
        def h(pos): return self.heuristic(pos, goal)
        self.open.append((h(start), start))
        # TODO: Implement full A* logic
        pass


def manhattan(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.sum(np.abs(p1 - p2)))


def main():
    maze = Maze()
    player = Player()
    maze.add_player(player)

    solver = AStarSolver(heuristic=manhattan)
    solver.solve(maze.maze, maze.init_pos, maze.shape - 3)

    maze.save_as_image(step=0, output_dir="outputs")


if __name__ == "__main__":
    main()
