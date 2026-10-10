import cv2
import numpy as np
import osmnx as ox
import networkx as nx
import gpxpy
import gpxpy.gpx
from datetime import datetime, timezone
from scipy.spatial.distance import directed_hausdorff

# -----------------------------------------------------------------------------
# STEP 1: IMAGE PROCESSING & VECTOR EXTRACTION
# -----------------------------------------------------------------------------
def extract_line_from_image(image_path: str, num_samples: int = 50) -> np.ndarray:
    """
    Loads an image containing a single continuous drawn line, extracts its 
    skeleton/contour, and samples 'num_samples' ordered (x, y) coordinates.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image file at {image_path}")

    # Threshold image to binary (assuming black line on light background)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours representing drawn stroke
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No drawn line detected in the image.")

    # Select largest continuous contour
    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.reshape(-1, 2)  # shape (N, 2) -> (x, y)

    # Downsample points evenly to standard length
    indices = np.linspace(0, len(points) - 1, num_samples, dtype=int)
    sampled_points = points[indices].astype(float)

    # Normalize coordinates to unit scale [0, 1]
    sampled_points -= sampled_points.min(axis=0)
    max_dim = sampled_points.max(axis=0).max()
    if max_dim > 0:
        sampled_points /= max_dim

    # Flip Y axis because pixel coordinates origin is top-left vs geographic bottom-left
    sampled_points[:, 1] = 1.0 - sampled_points[:, 1]

    return sampled_points


# -----------------------------------------------------------------------------
# STEP 2: SHAPE MATCHING & ROUTE FINDING
# -----------------------------------------------------------------------------
def compute_shape_similarity(route_coords: np.ndarray, target_shape: np.ndarray) -> float:
    """
    Calculates dissimilarity using symmetric Hausdorff distance between two 2D shapes.
    Lower score = closer visual match.
    """
    # Rescale both shapes to zero mean and unit variance for scale-invariant comparison
    r_norm = (route_coords - route_coords.mean(axis=0)) / (route_coords.std(axis=0) + 1e-6)
    t_norm = (target_shape - target_shape.mean(axis=0)) / (target_shape.std(axis=0) + 1e-6)

    d1 = directed_hausdorff(r_norm, t_norm)[0]
    d2 = directed_hausdorff(t_norm, r_norm)[0]
    return max(d1, d2)


def find_best_strava_art_location(
    city_name: str,
    target_shape: np.ndarray,
    target_miles: float,
    sample_nodes_limit: int = 30
) -> tuple:
    """
    Scans candidate street nodes in a city to locate a starting point and 
    street path that best matches the target shape.
    """
    print(f"1. Downloading street network for '{city_name}'...")
    G = ox.graph_from_place(city_name, network_type="walk")
    
    # Project graph to UTM meters for accurate distance calculations
    G_proj = ox.project_graph(G)
    nodes_df, _ = ox.graph_to_gdfs(G_proj)

    target_meters = target_miles * 1609.34
    print(f"2. Searching street sub-networks aiming for target distance ~{target_miles:.2f} miles ({target_meters:.0f} m)...")

    # Sample candidate start nodes across different neighborhoods
    candidate_nodes = list(G_proj.nodes())
    step = max(1, len(candidate_nodes) // sample_nodes_limit)
    sampled_starts = candidate_nodes[::step]

    best_score = float("inf")
    best_route_nodes = []

    num_points = len(target_shape)

    for i, start_node in enumerate(sampled_starts):
        # Scale shape points to real geographic meter offsets around candidate node
        start_x = G_proj.nodes[start_node]['x']
        start_y = G_proj.nodes[start_node]['y']

        # Scale target unit shape to match target total bounds distance
        scaled_shape = target_shape * (target_meters / 3.0)
        scaled_shape[:, 0] += start_x
        scaled_shape[:, 1] += start_y

        # Snap shape points to nearest real street nodes
        snapped_nodes = ox.distance.nearest_nodes(G_proj, X=scaled_shape[:, 0], Y=scaled_shape[:, 1])

        # Form continuous path connecting snapped nodes
        candidate_path = []
        possible = True
        for u, v in zip(snapped_nodes[:-1], snapped_nodes[1:]):
            try:
                sub_path = nx.shortest_path(G_proj, u, v, weight="length")
                if candidate_path:
                    candidate_path.extend(sub_path[1:])
                else:
                    candidate_path.extend(sub_path)
            except nx.NetworkXNoPath:
                possible = False
                break

        if not possible or len(candidate_path) < 2:
            continue

        # Extract generated path coordinates in projected space
        route_coords = np.array([(G_proj.nodes[n]['x'], G_proj.nodes[n]['y']) for n in candidate_path])

        # Evaluate similarity score
        score = compute_shape_similarity(route_coords, scaled_shape)

        if score < best_score:
            best_score = score
            best_route_nodes = candidate_path

    print(f"3. Best location match score: {best_score:.4f}")
    return G, best_route_nodes


# -----------------------------------------------------------------------------
# STEP 3: EXPORT TO GPX
# -----------------------------------------------------------------------------
def export_to_gpx(G, route_nodes: list, output_filename: str = "strava_matched_art.gpx"):
    """Converts selected graph nodes into a GPX file."""
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name="Strava Art Route")
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    for node in route_nodes:
        lat = G.nodes[node]['y']
        lon = G.nodes[node]['x']
        segment.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon, time=datetime.now(timezone.utc)))

    with open(output_filename, "w") as f:
        f.write(gpx.to_xml())

    print(f"Saved GPX route to '{output_filename}'")


# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # USER INPUTS
    CITY = "Seattle, King County, Washington, 98118, USA"
    TARGET_MILES = 8
    IMAGE_FILE = "leaf.png"  # Path to local sketch/line drawing

    # 1. Parse Image to vector points
    target_vector = extract_line_from_image(IMAGE_FILE, num_samples=50)

    # 2. Find best location in city matching shape & distance
    G, best_route = find_best_strava_art_location(CITY, target_vector, TARGET_MILES)

    # 3. Export route to GPX
    if best_route:
        export_to_gpx(G, best_route, "best_strava_art.gpx")
        # Visualize
        ox.plot_graph_route(G, best_route, route_color="r", route_linewidth=3, node_size=0)
    else:
        print("No valid route could be constructed.")