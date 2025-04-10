import numpy as np
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.basis.constant as ct

# Space parameters
SPACE_LENGTH = 1.6  # Space length in meters
SPACE_WIDTH = 1.7   # Space width in meters
SPACE_HEIGHT = 1.55  # Space height in meters
GRID_SIZE = 0.02    # Voxel size in meters
SPACE_Z_OFFSET = 0.365  # Z-axis offset for the space

# Noise point parameters
NOISE_POINTS_PER_OBSTACLE = 20  # Number of noise points around each obstacle
NOISE_MAX_DISTANCE = 0.02       # Maximum distance for noise points
NOISE_MIN_DISTANCE = 0.0        # Minimum distance for noise points
NOISE_SIZE = 0.01               # Size of noise points

def obstacle_setup(length, width, height, pos):
    """Create a collision model for an obstacle"""
    planecm = mcm.gen_box(xyz_lengths=[length, width, height],
                         pos=pos,
                         rgb=ct.chinese_red, alpha=1.)
    planecm.attach_to(base)
    return planecm

def generate_noise_around_obstacle(obstacle_pos, obstacle_size):
    """Generate random noise points tightly around obstacle surfaces"""
    noise_points = []

    for _ in range(NOISE_POINTS_PER_OBSTACLE):
        # Randomly select a face (0-5: front, back, left, right, top, bottom)
        face = np.random.randint(0, 6)

        # Generate a point on the surface
        offset = np.zeros(3)
        half_size = obstacle_size / 2

        if face == 0:  # +X (front)
            offset[0] = half_size[0] + NOISE_SIZE / 2  # just beyond the surface
            offset[1] = np.random.uniform(-half_size[1], half_size[1])
            offset[2] = np.random.uniform(-half_size[2], half_size[2])
        elif face == 1:  # -X (back)
            offset[0] = -half_size[0] - NOISE_SIZE / 2
            offset[1] = np.random.uniform(-half_size[1], half_size[1])
            offset[2] = np.random.uniform(-half_size[2], half_size[2])
        elif face == 2:  # +Y (left)
            offset[0] = np.random.uniform(-half_size[0], half_size[0])
            offset[1] = half_size[1] + NOISE_SIZE / 2
            offset[2] = np.random.uniform(-half_size[2], half_size[2])
        elif face == 3:  # -Y (right)
            offset[0] = np.random.uniform(-half_size[0], half_size[0])
            offset[1] = -half_size[1] - NOISE_SIZE / 2
            offset[2] = np.random.uniform(-half_size[2], half_size[2])
        elif face == 4:  # +Z (top)
            offset[0] = np.random.uniform(-half_size[0], half_size[0])
            offset[1] = np.random.uniform(-half_size[1], half_size[1])
            offset[2] = half_size[2] + NOISE_SIZE / 2
        elif face == 5:  # -Z (bottom)
            offset[0] = np.random.uniform(-half_size[0], half_size[0])
            offset[1] = np.random.uniform(-half_size[1], half_size[1])
            offset[2] = -half_size[2] - NOISE_SIZE / 2

        noise_pos = obstacle_pos + offset

        # Create and attach noise cube
        noise_point = obstacle_setup(
            length=NOISE_SIZE,
            width=NOISE_SIZE,
            height=NOISE_SIZE,
            pos=noise_pos
        )
        noise_points.append(noise_point)

    return noise_points


def generate_grid_obstacles(n_samples=10):
    """Generate grid-based obstacles with noise points"""
    # Calculate grid counts
    nx = int(SPACE_LENGTH / GRID_SIZE)
    ny = int(SPACE_WIDTH / GRID_SIZE)
    nz = int(SPACE_HEIGHT / GRID_SIZE)

    # Set maximum size limits
    max_length_grids = min(nx // 2, 15)
    max_width_grids = min(ny // 2, 15)
    max_height_grids = min(nz // 2, 10)

    obstacle_list = []
    all_noise_points = []
    
    for _ in range(n_samples):
        # Randomly generate obstacle dimensions (in grid units)
        length_grids = np.random.randint(1, max_length_grids)
        width_grids = np.random.randint(1, max_width_grids)
        height_grids = np.random.randint(1, max_height_grids)
        
        # Convert to actual dimensions
        length = length_grids * GRID_SIZE
        width = width_grids * GRID_SIZE
        height = height_grids * GRID_SIZE
        
        # Calculate valid position range
        max_x = (SPACE_LENGTH - length) / 2
        max_y = (SPACE_WIDTH - width) / 2
        max_z = SPACE_HEIGHT/2 - height + SPACE_Z_OFFSET
        min_z = -SPACE_HEIGHT/2 + SPACE_Z_OFFSET
        
        # Generate random position
        pos_x = np.random.uniform(-max_x, max_x)
        pos_y = np.random.uniform(-max_y, max_y)
        pos_z = np.random.uniform(min_z + height/2, max_z)
        
        pos = np.array([pos_x, pos_y, pos_z])
        
        # Create obstacle
        obstacle = obstacle_setup(length=length, width=width, height=height, pos=pos)
        obstacle_list.append(obstacle)
        
        # Generate noise points for obstacle
        noise_points = generate_noise_around_obstacle(pos, np.array([length, width, height]))
        all_noise_points.extend(noise_points)

    return obstacle_list + all_noise_points

def voxelize_scene(voxel_size=GRID_SIZE):
    """Convert scene obstacles into voxel representation"""
    # Scene boundaries (with z-offset)
    scene_min = np.array([-SPACE_LENGTH/2, -SPACE_WIDTH/2, -SPACE_HEIGHT/2 + SPACE_Z_OFFSET])
    scene_max = np.array([SPACE_LENGTH/2, SPACE_WIDTH/2, SPACE_HEIGHT/2 + SPACE_Z_OFFSET])
    
    # Create voxel grid
    voxel_dims = np.ceil((scene_max - scene_min) / voxel_size).astype(int)
    obstacle_voxels = np.zeros(voxel_dims, dtype=bool)
    
    # Collect all collision models
    for node in base.render.getChildren():
        collision_model = node.getPythonTag("collision_model")
        if collision_model is not None and isinstance(collision_model, mcm.CollisionModel):
            # Get model bounding box
            mesh = collision_model.trm_mesh
            mesh_min, mesh_max = mesh.bounds
            
            # Transform to world coordinates
            transform_mat = collision_model.pdndp.getMat()
            corners = np.array([
                transform_mat.xformPoint(Point3(mesh_min[0], mesh_min[1], mesh_min[2])),
                transform_mat.xformPoint(Point3(mesh_max[0], mesh_min[1], mesh_min[2])),
                transform_mat.xformPoint(Point3(mesh_min[0], mesh_max[1], mesh_min[2])),
                transform_mat.xformPoint(Point3(mesh_max[0], mesh_max[1], mesh_min[2])),
                transform_mat.xformPoint(Point3(mesh_min[0], mesh_min[1], mesh_max[2])),
                transform_mat.xformPoint(Point3(mesh_max[0], mesh_min[1], mesh_max[2])),
                transform_mat.xformPoint(Point3(mesh_min[0], mesh_max[1], mesh_max[2])),
                transform_mat.xformPoint(Point3(mesh_max[0], mesh_max[1], mesh_max[2]))
            ])
            
            # Calculate world-space bounding box
            box_min = np.maximum(np.min(corners, axis=0), scene_min)
            box_max = np.minimum(np.max(corners, axis=0), scene_max)
            
            # Convert to voxel indices
            start_idx = np.floor((box_min - scene_min) / voxel_size).astype(int)
            end_idx = np.ceil((box_max - scene_min) / voxel_size).astype(int)
            start_idx = np.maximum(start_idx, 0)
            end_idx = np.minimum(end_idx, voxel_dims)
            
            # Mark occupied voxels
            for x in range(start_idx[0], end_idx[0]):
                for y in range(start_idx[1], end_idx[1]):
                    for z in range(start_idx[2], end_idx[2]):
                        voxel_min = scene_min + np.array([x, y, z]) * voxel_size
                        voxel_max = voxel_min + voxel_size
                        if (voxel_max >= box_min).all() and (voxel_min <= box_max).all():
                            obstacle_voxels[x, y, z] = True
    
    return obstacle_voxels, scene_min

def visualize_voxels(voxels, voxel_origin, voxel_size):
    """Visualize voxel grid"""
    display_size = voxel_size * 0.85  # Display size slightly smaller than actual voxel
    
    if voxel_origin is not None and voxels is not None:
        # Get all occupied voxel positions
        occupied_indices = np.argwhere(voxels)
        
        for idx in occupied_indices:
            pos = voxel_origin + (idx * voxel_size + voxel_size/2)
            mgm.gen_box(xyz_lengths=[display_size] * 3,
                      pos=pos,
                      rgb=ct.red,  # Red for obstacles
                      alpha=1.).attach_to(base)


def visualize_space():
    """Visualize the entire space in the 3D world"""
    # Create a box representing the space boundaries
    space_box = mgm.gen_box(
        xyz_lengths=[SPACE_LENGTH, SPACE_WIDTH, SPACE_HEIGHT],
        pos=[0, 0, SPACE_Z_OFFSET],
        rgb=ct.blue,  # Set color to blue for space boundaries
        alpha=0.06     # Set transparency to make it visible without being too opaque
    )
    space_box.attach_to(base)


# Example usage
if __name__ == '__main__':
    # Initialize 3D visualization environment
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0.3])
    mgm.gen_frame().attach_to(base)  # Add reference frame
    visualize_space()

    import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
    from wrs import wd, rm, mcm
    robot_s = franka.FrankaResearch3(enable_cc=True)
    robot_s.goto_given_conf([0, 0, 0, 0, 0, 0, 0])
    robot_s.gen_meshmodel().attach_to(base)
    
    # 1. Generate random obstacles with noise points
    obstacles = generate_grid_obstacles(n_samples=6)
    
    # 2. Voxelize the scene
    obstacle_voxels, voxel_origin = voxelize_scene()
    
    # 3. Visualize voxels
    visualize_voxels(obstacle_voxels, voxel_origin, GRID_SIZE)
    
    # Run visualization
    base.run()