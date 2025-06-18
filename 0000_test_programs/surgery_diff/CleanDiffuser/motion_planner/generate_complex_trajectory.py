import numpy as np
import matplotlib.pyplot as plt

class ComplexTrajectoryGenerator:
    def __init__(self, num_segments=5, num_points_per_segment=100, scale_range=(0.1, 0.5), center_init=np.array([0.4, 0.2, 0.3])):
        """
        Initialize the trajectory generator with the specified parameters.
        
        Args:
            num_segments (int): Number of curve segments to combine.
            num_points_per_segment (int): Number of points per segment.
            scale_range (tuple): The range (min, max) for random scale values for each segment.
            center_init (np.array): Initial center for the first curve segment.
        """
        self.num_segments = num_segments
        self.num_points_per_segment = num_points_per_segment
        self.scale_range = scale_range
        self.center_init = center_init

    def generate_random_cubic_curve(self, num_points=20, scale=0.1, center=np.array([0.4, 0.2, 0.3])):
        """
        Generate a random cubic curve with a given center and scale.

        Args:
            num_points (int): Number of points to generate along the curve.
            scale (float): Scale factor for randomness.
            center (np.array): Center point of the curve in 3D space.

        Returns:
            np.array: Generated cubic curve points.
            np.array: The coefficients used to generate the curve.
        """
        a = np.random.randn(3) * scale
        b = np.random.randn(3) * scale
        c = np.random.randn(3) * scale
        d = center
        coeffs = np.hstack([a, b, c, d])  # Flatten to shape=(12,)

        t_vals = np.linspace(0, 1, num_points)
        points = [a * t**3 + b * t**2 + c * t + d for t in t_vals]

        return np.array(points), coeffs

    def generate(self):
        """
        Generate a complex trajectory by combining multiple cubic curves with random scales.
        
        Returns:
            np.array: Combined trajectory points.
        """
        all_points = []
        current_center = self.center_init

        for i in range(self.num_segments):
            # Randomly choose a scale for this segment from the range
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            
            # Generate random cubic curve for each segment
            points, _ = self.generate_random_cubic_curve(num_points=self.num_points_per_segment, scale=scale, center=current_center)
            
            # Append the points for the current segment
            all_points.append(points)
            
            # Update the center for the next segment to be the last point of the current segment
            current_center = points[-1]

        # Combine all the segments into one trajectory
        trajectory = np.vstack(all_points)

        return trajectory

    def plot(self):
        """
        Generate and plot the complex trajectory.
        """
        trajectory = self.generate()

        # Plot the complex trajectory
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Complex Trajectory')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Generated Complex 3D Trajectory with Random Scales')

        plt.show()

# Example usage:
if __name__ == "__main__":
    # Initialize the trajectory generator
    trajectory_generator = ComplexTrajectoryGenerator(num_segments=5, num_points_per_segment=100, scale_range=(0.1, 0.5), center_init=np.array([0.0, 0.0, 0.0]))

    # Plot the generated complex trajectory
    trajectory_generator.plot()
