import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PARAMETERS
# ==========================================================

# ALPHA is no longer used for routing. It is only used to
# report a weighted cost for the final route.
ALPHA = 0

BETA = 60     # (unused) Duplicated-edge preference (degrees)
GAMMA = 1000  # (unused) Bridge penalty


# ==========================================================
# GEOMETRY FUNCTIONS
# ==========================================================

def compute_polyline_length(pts):

    return np.sum(
        np.linalg.norm(
            pts[1:] - pts[:-1],
            axis=1
        )
    )


def edge_direction(pts):

    vec = pts[-1] - pts[0]

    norm = np.linalg.norm(vec)

    if norm < 1e-12:
        return np.array([1.0, 0.0])

    return vec / norm


def angle_between(v1, v2):

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-12 or n2 < 1e-12:
        return 0

    v1 = v1 / n1
    v2 = v2 / n2

    dot = np.clip(
        np.dot(v1, v2),
        -1.0,
        1.0
    )

    return np.arccos(dot)


def traversal_direction(pts, current_pos):

    d1 = np.linalg.norm(pts[0] - current_pos)
    d2 = np.linalg.norm(pts[-1] - current_pos)

    if d2 < d1:
        pts = pts[::-1]

    return edge_direction(pts)


# ==========================================================
# ROUTE METRICS
# ==========================================================

def compute_angle_weighted_cost(
    distance,
    total_angle,
    alpha=ALPHA
):

    theta = np.radians(total_angle)

    return (
        distance *
        (
            1 +
            alpha * np.sin(theta / 2)
        )
    )


def compute_total_turn_angle(E, route):

    total_angle = 0.0
    previous_dir = None

    for u, v, k in route:

        pts = E[u][v][k]["pts"]

        direction = traversal_direction(
            pts,
            E.nodes[u]["pos"]
        )

        if previous_dir is not None:

            total_angle += np.degrees(
                angle_between(
                    previous_dir,
                    direction
                )
            )

        previous_dir = direction

    return total_angle


# ==========================================================
# CPP FUNCTIONS
# ==========================================================

def build_cpp_graph(final_graph):

    G = nx.MultiGraph()

    for n, attrs in final_graph.nodes(data=True):
        G.add_node(n, **attrs)

    for u, v, k, data in final_graph.edges(
        keys=True,
        data=True
    ):

        attrs = dict(data)

        attrs["weight"] = compute_polyline_length(
            data["pts"]
        )

        G.add_edge(
            u,
            v,
            key=k,
            **attrs
        )

    return G


def find_odd_nodes(G):

    return [
        n
        for n in G.nodes
        if G.degree(n) % 2 == 1
    ]


# ==========================================================
# SHORTEST PATHS (DISTANCE ONLY)
# ==========================================================

def compute_odd_node_shortest_paths(
    G,
    odd_nodes
):

    shortest = {}

    for i, u in enumerate(odd_nodes):

        for v in odd_nodes[i + 1:]:

            path = nx.dijkstra_path(
                G,
                u,
                v,
                weight="weight"
            )

            distance = nx.dijkstra_path_length(
                G,
                u,
                v,
                weight="weight"
            )

            shortest[(u, v)] = {
                "distance": distance,
                "path": path
            }

    return shortest


# ==========================================================
# MINIMUM WEIGHT MATCHING (DISTANCE ONLY)
# ==========================================================

def minimum_weight_matching(
    odd_nodes,
    shortest
):

    matching_graph = nx.Graph()

    for (u, v), info in shortest.items():

        matching_graph.add_edge(
            u,
            v,
            weight=-info["distance"]
        )

    return nx.max_weight_matching(
        matching_graph,
        maxcardinality=True
    )


# ==========================================================
# EULERISE GRAPH
# ==========================================================

def eulerise_graph(
    G,
    matching,
    shortest
):

    E = G.copy()

    for u, v in matching:

        lookup = (u, v)

        if lookup not in shortest:
            lookup = (v, u)

        path = shortest[lookup]["path"]

        for a, b in zip(
            path[:-1],
            path[1:]
        ):

            edge_data = min(
                E.get_edge_data(a, b).values(),
                key=lambda d: d["weight"]
            )

            duplicated_attrs = dict(edge_data)

            duplicated_attrs["duplicated"] = True

            E.add_edge(
                a,
                b,
                **duplicated_attrs
            )

    return E


# ==========================================================
# EULER ROUTE (FLEURY + MINIMUM ANGLE DEFLECTION)
# ==========================================================

def build_euler_route(E, start=None):
    """
    Walk every edge of the Eulerian multigraph E exactly once.

    At each node, choose the untraversed edge that gives the
    smallest turn relative to the previous direction of travel.
    Bridges (in the remaining graph) are avoided unless they are
    the only option, which guarantees no edge is stranded.
    """

    R = E.copy()  # remaining (untraversed) edges

    if start is None:
        start = min(E.nodes)

    current = start
    previous_dir = None
    route = []

    def is_bridge(u, v, k):

        # Only one edge left at u: forced move, not a problem
        if R.degree(u) == 1:
            return False

        data = dict(R[u][v][k])

        R.remove_edge(u, v, key=k)
        connected = nx.has_path(R, u, v)
        R.add_edge(u, v, key=k, **data)

        return not connected

    while R.number_of_edges() > 0:

        candidates = []

        for _, nbr, k in R.edges(current, keys=True):

            pts = R[current][nbr][k]["pts"]

            direction = traversal_direction(
                pts,
                E.nodes[current]["pos"]
            )

            if previous_dir is None:
                deflection = 0.0
            else:
                deflection = angle_between(
                    previous_dir,
                    direction
                )

            candidates.append(
                (deflection, nbr, k, direction)
            )

        safe = [
            c for c in candidates
            if not is_bridge(current, c[1], c[2])
        ]

        pool = safe if safe else candidates

        deflection, nbr, k, direction = min(
            pool,
            key=lambda c: c[0]
        )

        route.append((current, nbr, k))

        R.remove_edge(current, nbr, key=k)

        previous_dir = direction
        current = nbr

    return route


# ==========================================================
# POLYLINE BUILDING
# ==========================================================

def route_to_polyline(
    E,
    route
):

    drawing_pts = []

    for u, v, k in route:

        pts = E[u][v][k]["pts"].copy()

        if len(drawing_pts) == 0:

            if np.linalg.norm(
                pts[-1] - E.nodes[u]["pos"]
            ) < np.linalg.norm(
                pts[0] - E.nodes[u]["pos"]
            ):
                pts = pts[::-1]

            drawing_pts.extend(pts)

            continue

        start_dist = np.linalg.norm(
            drawing_pts[-1] - pts[0]
        )

        end_dist = np.linalg.norm(
            drawing_pts[-1] - pts[-1]
        )

        if end_dist < start_dist:
            pts = pts[::-1]

        drawing_pts.extend(
            pts[1:]
        )

    return np.array(drawing_pts)


# ==========================================================
# CHINESE POSTMAN
# ==========================================================

def chinese_postman(final_graph):

    print("Building weighted graph...")

    G = build_cpp_graph(final_graph)

    odd_nodes = find_odd_nodes(G)

    print(
        f"Odd vertices found: {len(odd_nodes)}"
    )

    if len(odd_nodes) == 0:

        E = G

    else:

        print("Finding shortest paths (distance only)...")

        shortest = compute_odd_node_shortest_paths(
            G,
            odd_nodes
        )

        for (u, v), info in shortest.items():

            print(
                f"Path {u} -> {v}: "
                f"Length = {info['distance']:.2f}"
            )

        print("Solving minimum weight matching...")

        matching = minimum_weight_matching(
            odd_nodes,
            shortest
        )

        print(
            f"Matched pairs: {len(matching)}"
        )

        print("Eulerising graph...")

        E = eulerise_graph(
            G,
            matching,
            shortest
        )

    print("Computing angle-guided Euler circuit...")

    route = build_euler_route(E)

    polyline = route_to_polyline(
        E,
        route
    )

    return {
        "euler_graph": E,
        "route": route,
        "polyline": polyline
    }


# ==========================================================
# PLOTTING
# ==========================================================

def plot_original_graph(G):

    plt.figure(figsize=(8,6))

    for _, _, data in G.edges(data=True):

        pts = data["pts"]

        plt.plot(
            pts[:,0],
            pts[:,1],
            'k-',
            linewidth=2
        )

    for n, attrs in G.nodes(data=True):

        x, y = attrs["pos"]

        plt.scatter(
            x,
            y,
            color="blue",
            s=100
        )

        plt.text(
            x+0.1,
            y+0.1,
            str(n)
        )

    plt.title("Original Graph")
    plt.axis("equal")
    plt.show()


def plot_eulerised_graph(E):

    plt.figure(figsize=(8,6))

    for _, _, _, data in E.edges(
        keys=True,
        data=True
    ):

        pts = data["pts"]

        if data.get(
            "duplicated",
            False
        ):

            plt.plot(
                pts[:,0],
                pts[:,1],
                color="red",
                linewidth=4
            )

        else:

            plt.plot(
                pts[:,0],
                pts[:,1],
                color="black",
                linewidth=2
            )

    for n, attrs in E.nodes(data=True):

        x, y = attrs["pos"]

        plt.scatter(
            x,
            y,
            color="blue",
            s=100
        )

        plt.text(
            x+0.1,
            y+0.1,
            str(n)
        )

    plt.title(
        "Eulerised Graph\n(Red = CPP Duplicated Edges)"
    )

    plt.axis("equal")
    plt.show()


def plot_route(E, route):

    plt.figure(figsize=(10,8))

    cmap = plt.cm.viridis

    N = len(route)

    for i, (u, v, k) in enumerate(route):

        pts = E[u][v][k]["pts"].copy()

        if np.linalg.norm(
            pts[-1] - E.nodes[u]["pos"]
        ) < np.linalg.norm(
            pts[0] - E.nodes[u]["pos"]
        ):
            pts = pts[::-1]

        colour = cmap(i / max(N,1))

        plt.plot(
            pts[:,0],
            pts[:,1],
            color=colour,
            linewidth=3
        )

        midpoint = pts.mean(axis=0)

        plt.text(
            midpoint[0],
            midpoint[1],
            str(i),
            fontsize=8
        )

    for n, attrs in E.nodes(data=True):

        x, y = attrs["pos"]

        plt.scatter(
            x,
            y,
            color="red",
            s=70
        )

    plt.title(
        "Euler Traversal Order"
    )

    plt.axis("equal")
    plt.show()


# ==========================================================
# TEST GRAPH
# ==========================================================

final_graph = nx.MultiGraph()

positions = {
    0:(0,0),
    1:(2,0),
    2:(4,0),
    3:(6,0),
    4:(1,2),
    5:(3,2),
    6:(5,2),
    7:(7,2),
    8:(0,4),
    9:(2,4),
    10:(4,4),
    11:(6,4)
}

for n, pos in positions.items():

    final_graph.add_node(
        n,
        pos=np.array(
            pos,
            dtype=float
        )
    )


def add_edge(a,b):

    pts = np.array([
        positions[a],
        positions[b]
    ], dtype=float)

    final_graph.add_edge(
        a,
        b,
        pts=pts
    )


add_edge(0,1)
add_edge(1,2)
add_edge(2,3)

add_edge(4,5)
add_edge(5,6)
add_edge(6,7)

add_edge(8,9)
add_edge(9,10)
add_edge(10,11)

add_edge(0,4)
add_edge(1,5)
add_edge(2,6)
add_edge(3,7)

add_edge(4,8)
add_edge(5,9)
add_edge(6,10)
add_edge(7,11)

add_edge(0,5)
add_edge(2,5)
add_edge(2,7)

add_edge(4,9)
add_edge(6,9)
add_edge(6,11)


# ==========================================================
# RUN
# ==========================================================

result = chinese_postman(
    final_graph
)

E = result["euler_graph"]
route = result["route"]

print(
    "\nEuler graph edges:",
    E.number_of_edges()
)

print(
    "Route edges:",
    len(route)
)

print(
    "Graph Eulerian:",
    nx.is_eulerian(E)
)

# ----------------------------------------------------------
# Completeness checks: every edge used exactly once
# ----------------------------------------------------------

assert len(route) == E.number_of_edges(), \
    "Some edges were missed!"

assert len(
    {(min(u, v), max(u, v), k) for u, v, k in route}
) == len(route), \
    "Some edges were traversed more than once!"

print("All edges traversed exactly once: OK")

turning_angle = compute_total_turn_angle(
    E,
    route
)

total_distance = sum(
    E[u][v][k]["weight"]
    for u, v, k in route
)

weighted_route_cost = compute_angle_weighted_cost(
    total_distance,
    turning_angle
)

print(
    f"Total route distance: {total_distance:.2f}"
)

print(
    f"Total turning angle: {turning_angle:.2f} degrees"
)

print(
    f"Alpha: {ALPHA}"
)

print(
    f"Weighted route cost: {weighted_route_cost:.2f}"
)

plot_original_graph(final_graph)
plot_eulerised_graph(E)
plot_route(E, route)