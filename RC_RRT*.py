import math
import statistics
import random
import time
import numpy as np
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt

# =============================================================================================================================
def is_occupied(M, s):
    """Return whether a sample vertex is within bounds and occupied"""
    c, r = s
    if 0 <= r < M.shape[0] and 0 <= c < M.shape[1]:
        return M[r][c] == 0
    return False

def is_free(M, s):
    """Return whether a sample vertex is within bounds and occupied"""
    c, r = s
    if 0 <= r < M.shape[0] and 0 <= c < M.shape[1]:
        return M[r][c] == 1
    return False

def sample_free(M, N=1000):
    """Generate N samples uniformly randomly in free space of M"""
    rows, cols = M.shape
    free_points = []
    while len(free_points) < N:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if M[r][c] == 1: # if free
            free_points.append((c, r))
    return free_points

def bresenham(v1, v2):
    """Bresenham: generates a list of points along the edge connecting to vertices"""
    x1, y1 = v1
    x2, y2 = v2
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append((round(x1), round(y1)))
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

def sample_near_edges(corner_points, edge_radius=10, N=5):
    """Sample N points around the edges formed by corner points."""
    sample_points = []
    # Generate sample points around each edge formed by all other corner points
    for i in range(len(corner_points)):
        p1 = corner_points[i]
        for j in range(1, len(corner_points)):
            p2 = corner_points[(i + j) % len(corner_points)]
            for _ in range(N):
                t = random.random()  # t goes from 0 (p1) to 1 (p2)
                x = round(p1[0] + t * (p2[0] - p1[0]) + random.uniform(-edge_radius, edge_radius))
                y = round(p1[1] + t * (p2[1] - p1[1]) + random.uniform(-edge_radius, edge_radius))
                if is_free(M, (x, y)):
                    sample_points.append((x, y))
    return sample_points

def sample_radius(center, radius, N=20):
    """Sample N samples around radius of a point"""
    radians = np.linspace(0, 2*np.pi, N)
    points = []
    for angle in radians:
        x = round(center[0] + radius*math.cos(angle))
        y = round(center[1] + radius*math.sin(angle))
        points.append((x, y))
    return points

def is_corner_extend (M, center, vector, radius, distance):
    """Return whether a point is deemed a corner - DO NOT USE"""
    # extend along vector in both directions to obtain auxilary points
    # radius = math.sqrt(vector[0]**2 + vector[1]**2)
    dx1 = vector[0] * (radius - distance) / radius
    dy1 = vector[1] * (radius - distance) / radius
    px1 = round(center[0] + dx1)
    py1 = round(center[1] + dy1)
    dx2 = vector[0] * (radius + distance) / radius
    dy2 = vector[1] * (radius + distance) / radius
    px2 = round(center[0] + dx2)
    py2 = round(center[1] + dy2)
    # deemed a corner if at least one of the points is not obstructed
    if is_occupied(M, (px1, py1)) and is_occupied(M, (px2, py2)):
        return False
    return True

def is_corner(M, center, vector, radius, distance):
    """Return whether a point is deemed a corner"""
    # extend tangenial to vector in both directions to obtain auxilary points
    # radius = math.sqrt(vector[0]**2 + vector[1]**2)
    dx = vector[1] * distance / radius
    dy = vector[0] * distance / radius
    px1 = round(center[0] + vector[0] + dx)
    py1 = round(center[1] + vector[1] - dy)
    px2 = round(center[0] + vector[0] - dx)
    py2 = round(center[1] + vector[1] + dy)
    # deemed a corner if both points are not obstructed
    if is_free(M, (px1, py1)) and is_free(M, (px2, py2)):
        return True
    return False

def form_vector(s, points):
    """Form a vector for each sample vertex"""
    radial_vectors = []
    for p in points:
        if is_occupied(M, p):
            radial_vectors.append((p[0] - s[0], p[1] - s[1]))
    if len(radial_vectors) > 0:
        x, y = zip(*radial_vectors)
        vx = statistics.mean(x)
        vy = statistics.mean(y)
        return (vx, vy)
    return None

def find_corners(M, S, radius=10, N=20, distance=5):
    """Find the obstacle corner points in a sample set"""
    vectors = []
    corner_points = []
    for s in S:
        # sample points around radius
        points = sample_radius(s, radius, N)
        # get obstructed vector
        vector = form_vector(s, points)
        if vector == None:
            continue
        # test whether the sample vertex is deemed a corner
        if is_corner(M, s, vector, radius, distance):
            vectors.append(vector)
            corner_points.append(s)
    return vectors, corner_points

# ==============================================================================================================================
# Node class only used in RRT classes
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0



# RRT* algorithm
class RRTStar:
    def __init__(self, M, start, goal, step_size=20.0, search_radius=50.0, goal_radius=30.0, goal_sampling_probability=0.1, max_iterations=5000):
        self.map = M
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.step_size = step_size
        self.search_radius = search_radius
        self.goal_radius = goal_radius
        self.goal_sampling_probability = goal_sampling_probability
        self.max_iterations = max_iterations
        self.node_list = [self.start]
        self.path = None
        self.goal_reached = False

    def plan(self):
        """RRT* loop"""
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
            
    def sample_free(self):
        """Sample a node from free space"""
        if random.random() < self.goal_sampling_probability:
            # Choose the goal
            return self.goal
        s = sample_free(self.map, 1)[0]
        return Node(s[0], s[1])
    
    def nearest(self, node_list, rand_node):
        """Find the nearest node to the random node in the tree"""
        distances = [np.linalg.norm([node.x - rand_node.x, node.y - rand_node.y]) for node in node_list]
        nearest_node = node_list[np.argmin(distances)]
        return nearest_node
    
    def steer(self, from_node, to_node):
        """Create a new node along the heading from the parent node to the random node"""
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_node = Node(round(from_node.x + self.step_size * math.cos(theta)), round(from_node.y + self.step_size * math.sin(theta)))
        new_node.cost = from_node.cost + self.step_size
        new_node.parent = from_node
        return new_node

    def is_obstacle_free(self, node):
        """Check if a node is in free space"""
        return is_free(self.map, (node.x, node.y)) # Note: will return True if outside map bounds
    
    def near(self, new_node):
        """Find nodes within the search radius"""
        return [node for node in self.node_list if np.linalg.norm([node.x - new_node.x, node.y - new_node.y]) < self.search_radius]

    def is_collision_free(self, node1, node2):
        """Check if the edge connecting two nodes is in free space"""
        points = bresenham((node1.x, node1.y), (node2.x, node2.y))
        for point in points:
            if is_occupied(self.map, point):
                return False
        return True

    def form_edge(self, near_nodes, nearest_node, new_node):
        """Connect along minimum cost path"""
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
        """Rewire the tree by checking if any near nodes should adopt the new node as a parent"""
        for near_node in near_nodes:
            cost = new_node.cost + np.linalg.norm([near_node.x - new_node.x, near_node.y - new_node.y])
            if self.is_collision_free(near_node, new_node) and cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = cost

    def reached_goal(self, node):
        """Check if goal has been reached"""
        return np.linalg.norm([node.x - self.goal.x, node.y - self.goal.y]) < self.goal_radius

    def generate_path(self, goal_node):
        """Generate path from start to goal"""
        path = []
        node = goal_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]
    
    def to_points(self):
        return [(node.x, node.y) for node in self.node_list]



# RCRRT* algorithm
class RCRRTStar(RRTStar):
    def __init__(self, M, start, goal, corner_points, step_size=10.0, search_radius=50.0, goal_radius=30.0, goal_sampling_probability=0.1, max_iterations=5000, edge_radius=10, corner_sampling_probability=0.4):
        # Initialize RRTStar base class
        super().__init__(M, start, goal, step_size, search_radius, goal_radius, goal_sampling_probability, max_iterations)
        # Initialize other parameters
        self.corner_points = corner_points
        self.edge_points = sample_near_edges(corner_points + [start], edge_radius, N=2)
        self.corner_sampling_probability = corner_sampling_probability

    def sample_free(self):
        """Override sample_free method to use region-constrained sampling"""
        sampling_probability = random.random()
        if sampling_probability < self.goal_sampling_probability:
            # Choose the goal
            return self.goal
        if sampling_probability < self.corner_sampling_probability + self.goal_sampling_probability:
            # Choose a corner
            if self.corner_points:
                s = random.choice(self.corner_points)
                return Node(s[0], s[1])
        # Otherwise, choose a point near an edge...
        # if self.edge_points:
        #     s = random.choice(self.edge_points)
        #     return Node(s[0], s[1])

        # ...OR if no edges, sample uniformly
        s = sample_free(self.map, 1)[0]        
        return Node(s[0], s[1])



# ==============================================================================================================================
def plot_path(M, path, ax=None, size=1, color='tab:blue'):
    """Plot map and path"""
    if not path:
        print("Invalid path: cannot display.")
        return
    # Plot occupancy grid
    M = np.array(M)
    ax.imshow(M, cmap='gray')
    # Plot optimal path
    path_c, path_r  = zip(*path)
    ax.plot(path_c, path_r, marker='o', markersize=size, color=color, alpha=0.5)

def plot_points(M, points, ax=None, size=1, color='tab:orange'):
    """Plot map and points"""
    if not points:
        print("Invalid points: cannot display.")
        return
    # Plot map
    M = np.array(M)
    ax.imshow(M, cmap='gray')
    # Plot points
    c, r = zip(*points)
    ax.scatter(c, r, marker='o', s=size, color=color, alpha=0.5)

def plot_tree(M, tree,ax=None, size=2, color='tab:brown'):
    """Plot map and RRT* tree"""
    if not tree:
        print("Invalid tree: cannot display.")
        return

    # Convert the map to numpy array
    M = np.array(M)

    # Create a plot
    ax.imshow(M, cmap='gray')

    # Plot edges (lines between parent-child nodes)
    for node in tree:
        if node.parent is not None:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], color=color, alpha=0.5)

    # Plot nodes
    nodes_x = [node.x for node in tree]
    nodes_y = [node.y for node in tree]
    ax.scatter(nodes_x, nodes_y, s=size, color=color)
    
def load_occupancy_grid(image_path, threshold=1):
    """Load occupancy grid from PNG"""
    img = Image.open(image_path).convert('L')  # Convert image to grayscale
    binary_grid = (np.asarray(img) > threshold).astype(int) # Threshold to convert to binary graph
    return binary_grid

def calculate_path_length(path):
    """Calculate the total length of a path given a list of points"""
    length = 0
    for i in range(1, len(path)):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return length

# MAIN ===========================================================================================================================
NUM_TRIALS = 3
CORNER_RADIUS = 30
CORNER_N = 20
CORNER_DIST = 30

# Load Map
M = load_occupancy_grid("maps/junction.png")
start = sample_free(M,1)[0]
goal = sample_free(M,1)[0]

# Setup best trial
best_rrt_star_time = float('inf')
best_rcrrt_star_time = float('inf')
best_rrt_star_path_length = float('inf')
best_rcrrt_star_path_length = float('inf')

# Setup cumulative variables for averages
total_rrt_star_time = 0
total_rcrrt_star_time = 0
total_rrt_star_path_length = 0
total_rcrrt_star_path_length = 0
total_pre_processing_time = 0

successful_rrt_star_trials = 0
successful_rcrrt_star_trials = 0

# Perform many trials and store best of each algorithm
for trial in range(NUM_TRIALS):
    print(f'Trial {trial + 1:d} / {NUM_TRIALS:d}')
    
    # RRT*
    rrt_star = RRTStar(M, start, goal, step_size=30.0, search_radius=50.0, goal_radius=30.0, goal_sampling_probability=0.5, max_iterations=20000)
    start_time = time.time()
    rrt_star.plan()
    rrt_star_time = time.time() - start_time
    total_rrt_star_time += rrt_star_time
    rrt_star_path_length = float('inf')

    if rrt_star.path:
        rrt_star_path_length = calculate_path_length(rrt_star.path)
        total_rrt_star_path_length += rrt_star_path_length
        successful_rrt_star_trials += 1

    # Update best for RRT*
    if rrt_star_time < best_rrt_star_time and rrt_star.path:
        best_rrt_star_time = rrt_star_time
        best_rrt_star = rrt_star
        best_rrt_star_path_length = rrt_star_path_length

    # PRE-PROCESSING
    start_time = time.time()
    S = sample_free(M, 2000)
    V, C = find_corners(M, S, radius=CORNER_RADIUS, N=CORNER_N, distance=CORNER_DIST)
    pre_processing_time = time.time() - start_time
    total_pre_processing_time += pre_processing_time

    # RC-RRT*
    rcrrt_star = RCRRTStar(M, start, goal, C, step_size=30.0, search_radius=50.0, goal_radius=30.0, goal_sampling_probability=0.5, edge_radius=20, corner_sampling_probability=0.3, max_iterations=20000)
    start_time = time.time()
    rcrrt_star.plan()
    rcrrt_star_time = time.time() - start_time
    total_rcrrt_star_time += rcrrt_star_time
    rcrrt_star_path_length = float('inf')  

    if rcrrt_star.path:
        rcrrt_star_path_length = calculate_path_length(rcrrt_star.path)
        total_rcrrt_star_path_length += rcrrt_star_path_length
        successful_rcrrt_star_trials += 1

    # Update best for RC-RRT*
    if rcrrt_star_time < best_rcrrt_star_time and rcrrt_star.path:
        best_rcrrt_star_time = rcrrt_star_time
        best_rcrrt_star = rcrrt_star
        best_rcrrt_star_path_length = rcrrt_star_path_length
        best_S = S
        best_V = V
        best_C = C

# Calculate averages
avg_rrt_star_time = total_rrt_star_time / NUM_TRIALS
avg_rcrrt_star_time = total_rcrrt_star_time / NUM_TRIALS
avg_rrt_star_path_length = total_rrt_star_path_length / successful_rrt_star_trials if successful_rrt_star_trials > 0 else np.nan
avg_rcrrt_star_path_length = total_rcrrt_star_path_length / successful_rcrrt_star_trials if successful_rcrrt_star_trials > 0 else np.nan
avg_pre_processing_time = total_pre_processing_time / NUM_TRIALS

# Print results
print(f"Average RRT* Time: {avg_rrt_star_time:.2f} s")
print(f"Average RRT* Path Length: {avg_rrt_star_path_length:.2f} pixels")
print(f"Average RC-RRT* Time: {avg_rcrrt_star_time:.2f} s")
print(f"Average RC-RRT* Path Length: {avg_rcrrt_star_path_length:.2f} pixels")
print(f"Average Pre-Processing Time: {avg_pre_processing_time:.2f} s")
print(f"Successful RRT* Trials: {successful_rrt_star_trials} / 50")
print(f"Successful RCRRT* Trials: {successful_rcrrt_star_trials} / 50")

# Plot (same as before)
fig, axes = plt.subplots(2, 2, figsize=(10, 5))

# Plot best RRT*
ax1 = axes[0, 0]
plot_tree(M, best_rrt_star.node_list, ax=ax1)
plot_path(M, best_rrt_star.path, ax=ax1)
plot_points(M, [start, goal], ax=ax1, size=5)
ax1.set_title(f'RRT* - Time: {best_rrt_star_time:.2f} s, Length: {best_rrt_star_path_length:.2f} pixels')
ax1.axis('off')

# Plot best RC-RRT*
ax2 = axes[0, 1]
plot_points(M, best_C, ax=ax2, size=2, color='tab:red')
plot_tree(M, best_rcrrt_star.node_list, ax=ax2)
plot_path(M, best_rcrrt_star.path, ax=ax2)
plot_points(M, [start, goal], ax=ax2, size=5)
ax2.set_title(f'RC-RRT* - Time: {best_rcrrt_star_time:.2f} + {avg_pre_processing_time:.2f} s, Length: {best_rcrrt_star_path_length:.2f} pixels')
ax2.axis('off')

# Plot best Pre-Processing
ax3 = axes[1, 1]
plot_points(M, best_S, ax=ax3, color='tab:blue')
plot_points(M, best_C, ax=ax3, size=2, color='tab:red')
corner_radii = []
for corner in best_C:
    corner_radii += sample_radius(corner, CORNER_RADIUS, CORNER_N)
plot_points(M, corner_radii, ax=ax3, size=0.5, color='tab:gray')
v_points = [[best_V[i][j] + best_C[i][j] for j in range(2)] for i in range(min(len(best_V), len(best_C)))]
plot_points(M, v_points, ax=ax3, color='tab:orange')
ax3.set_title(f'Pre-Sampled Corners - Time: {avg_pre_processing_time:.2f} s')
ax3.axis('off')

fig.delaxes(axes[1, 0])
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.tight_layout()
plt.show()