import math
import random
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# =============================================================================================================================
# Utility functions
def is_occupied(M, s):
    c, r = s
    if 0 <= r < M.shape[0] and 0 <= c < M.shape[1]:
        return M[r][c] == 0
    return False

def is_free(M, s):
    c, r = s
    if 0 <= r < M.shape[0] and 0 <= c < M.shape[1]:
        return M[r][c] == 1
    return False

def sample_free(M, N=1000):
    rows, cols = M.shape
    free_points = []
    while len(free_points) < N:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if M[r][c] == 1:  # Free space
            free_points.append((c, r))
    return free_points

def bresenham(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

def load_occupancy_grid(image_path, threshold=128):
    img = Image.open(image_path).convert('L')  # Grayscale
    binary_grid = (np.asarray(img) > threshold).astype(int)  # Binarize
    return binary_grid

# ==============================================================================================================================
# Node class for RRT
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0

# ==============================================================================================================================
# RRT* class with dynamic replanning support
class RRTStar:
    def __init__(self, M, start, goal, step_size=20.0, search_radius=50.0, goal_radius=30.0, max_iterations=5000):
        self.map = M
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.step_size = step_size
        self.search_radius = search_radius
        self.goal_radius = goal_radius
        self.max_iterations = max_iterations
        self.node_list = [self.start]
        self.path = None
        self.goal_reached = False

    def plan(self):
        for _ in range(self.max_iterations):
            rand_node = self.sample_free()
            nearest_node = self.nearest(self.node_list, rand_node)
            new_node = self.steer(nearest_node, rand_node)

            if self.is_obstacle_free(new_node):
                near_nodes = self.near(new_node)
                new_node = self.form_edge(near_nodes, nearest_node, new_node)
                self.node_list.append(new_node)
                self.rewire(new_node, near_nodes)

            if self.reached_goal(new_node):
                self.path = self.generate_path(new_node)
                self.goal_reached = True
                return

    def plan_from_current(self, current_position):
        self.start = Node(current_position[0], current_position[1])
        self.node_list = [self.start]
        self.plan()

    def sample_free(self):
        v = sample_free(self.map, 1)[0]
        return Node(v[0], v[1])

    def nearest(self, node_list, rand_node):
        distances = [np.linalg.norm([node.x - rand_node.x, node.y - rand_node.y]) for node in node_list]
        return node_list[np.argmin(distances)]

    def steer(self, from_node, to_node):
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node = Node(round(from_node.x + self.step_size * math.cos(theta)),
                        round(from_node.y + self.step_size * math.sin(theta)))
        new_node.cost = from_node.cost + self.step_size
        new_node.parent = from_node
        return new_node

    def is_obstacle_free(self, node):
        return is_free(self.map, (node.x, node.y))

    def near(self, new_node):
        return [node for node in self.node_list if np.linalg.norm([node.x - new_node.x, node.y - new_node.y]) < self.search_radius]

    def is_collision_free(self, node1, node2):
        points = bresenham((node1.x, node1.y), (node2.x, node2.y))
        return all(is_free(self.map, p) for p in points)

    def form_edge(self, near_nodes, nearest_node, new_node):
        min_node = nearest_node
        min_cost = nearest_node.cost + np.linalg.norm([new_node.x - nearest_node.x, new_node.y - nearest_node.y])
        for near_node in near_nodes:
            cost = near_node.cost + np.linalg.norm([new_node.x - near_node.x, new_node.y - near_node.y])
            if self.is_collision_free(near_node, new_node) and cost < min_cost:
                min_node = near_node
                min_cost = cost
        new_node.cost = min_cost
        new_node.parent = min_node
        return new_node

    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            cost = new_node.cost + np.linalg.norm([near_node.x - new_node.x, near_node.y - new_node.y])
            if self.is_collision_free(new_node, near_node) and cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = cost

    def reached_goal(self, node):
        return np.linalg.norm([node.x - self.goal.x, node.y - self.goal.y]) < self.goal_radius

    def generate_path(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]

# ==============================================================================================================================
# Robot and navigation
class Robot:
    def __init__(self, position):
        self.position = position

    def move_to(self, position):
        self.position = position
        time.sleep(0.5)

    def get_position(self):
        return self.position

def detect_obstacles(robot_position, detection_range=10):
    x, y = robot_position
    return [(x + random.randint(-detection_range, detection_range), y + random.randint(-detection_range, detection_range))]

def update_obstacles(M, obstacles):
    for x, y in obstacles:
        if 0 <= y < M.shape[0] and 0 <= x < M.shape[1]:
            M[y][x] = 0

def visualize_map(M, current_path=None, tree=None, obstacles=None, robot_position=None, goal_position=None, all_paths=None):
    """Visualize the map, RRT* tree, all paths, current path, obstacles, and robot."""
    plt.clf()
    plt.imshow(M, cmap='gray', origin='upper')

    # Plot the RRT* tree (nodes and edges)
    if tree:
        for node in tree:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color='brown', linewidth=0.5)  # Edges
        # Plot all nodes
        nodes_x = [node.x for node in tree]
        nodes_y = [node.y for node in tree]
        plt.scatter(nodes_x, nodes_y, color='orange', s=5, label='RRT* Nodes')

    # Plot all previous paths
    if all_paths:
        for path in all_paths[:-1]:  # Plot old paths in a lighter color
            path_x, path_y = zip(*path)
            plt.plot(path_x, path_y, 'c--', linewidth=1, alpha=0.6, label='Previous Paths')

    # Plot the current path
    if current_path:
        path_x, path_y = zip(*current_path)
        plt.plot(path_x, path_y, 'b-o', markersize=3, label='Current Path')

    # Plot dynamic obstacles
    if obstacles:
        for obs in obstacles:
            plt.plot(obs[0], obs[1], 'rx', markersize=5, label='Obstacle')

    # Plot the robot's current position
    if robot_position:
        plt.plot(robot_position[0], robot_position[1], 'go', label='Robot', markersize=8)

    # Plot the goal
    if goal_position:
        plt.plot(goal_position[0], goal_position[1], 'mo', label='Goal', markersize=8)

    plt.legend(loc='upper right')
    plt.pause(0.1)


def navigate_with_visualization(robot, planner, goal, M):
    """Navigate the robot along a path with visualization and dynamic replanning."""
    all_paths = [planner.path]  # Store all paths taken
    obstacles = []
    
    current_path = planner.path
    for waypoint in current_path:
        # Move the robot to the next waypoint
        robot.move_to(waypoint)
        robot_position = robot.get_position()

        # Detect and add new obstacles
        new_obstacles = detect_obstacles(robot_position)
        if new_obstacles:
            obstacles.extend(new_obstacles)
            update_obstacles(M, new_obstacles)

            # Replan from the robot's current position
            planner.plan_from_current(robot_position)
            if planner.path:
                all_paths.append(planner.path)  # Track the new path
                current_path = planner.path
            else:
                print("No valid path found!")
                break

        # Visualize the current state, showing all paths
        visualize_map(M, current_path, planner.node_list, obstacles, robot_position, goal, all_paths)

        # Check if the goal is reached
        if robot_position == goal:
            print("Goal reached!")
            break


# ==============================================================================================================================
# Main execution with visualization
M = load_occupancy_grid("maps/shapes.png")
start = sample_free(M, 1)[0]
goal = sample_free(M, 1)[0]
robot = Robot(start)

planner = RRTStar(M, start, goal)
planner.plan()

if planner.path:
    print(f"Initial path: {planner.path}")
    plt.figure(figsize=(10, 10))
    navigate_with_visualization(robot, planner, goal, M)
else:
    print("No initial path found.")

