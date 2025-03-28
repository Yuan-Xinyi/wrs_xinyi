# please install the ruckig library first: pip install ruckig
from ruckig import InputParameter, Ruckig, Trajectory, Result
import numpy as np
import matplotlib.pyplot as plt

# import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm_s
import wrs.robot_sim.manipulators.franka_research_3_arm.franka_research_3_arm as franka
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm

 
def gen_collision_free_start_goal(robot):
    '''generate the start and goal conf'''
    while True:
        start_conf = robot.rand_conf()
        goal_conf = robot.rand_conf()
        robot.goto_given_conf(jnt_values=start_conf)
        start_cc = robot.cc.is_collided()
        robot.goto_given_conf(jnt_values=goal_conf)
        goal_cc = robot.cc.is_collided()
        if not start_cc and not goal_cc:
            break
    return start_conf, goal_conf


if __name__ == '__main__':
    '''init the robot and world'''
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = franka.FrankaResearch3Arm(enable_cc=True)
    inp = InputParameter(robot.n_dof)

    '''assign the default dynamic constraints'''
    start_conf, goal_conf = gen_collision_free_start_goal(robot)
    inp.current_position, inp.target_position = start_conf, goal_conf
    inp.current_velocity = rm.np.zeros(robot.n_dof) # support non-zero vel, zero for simplicity
    inp.current_acceleration = rm.np.zeros(robot.n_dof) # support non-zero acc, zero for simplicity
    inp.target_velocity = rm.np.zeros(robot.n_dof)
    inp.target_acceleration = rm.np.zeros(robot.n_dof)

    inp.max_velocity = rm.np.asarray([rm.pi * 2 / 3] * robot.n_dof)
    inp.max_acceleration = rm.np.asarray([rm.pi] * robot.n_dof)
    inp.max_jerk = rm.np.asarray([rm.pi * 2] * robot.n_dof)

 
    # Set different constraints for negative direction
    inp.min_velocity = [-value for value in inp.max_velocity]
    inp.min_acceleration = [-value for value in inp.max_acceleration]
 
    # We don't need to pass the control rate (cycle time) when using only offline features
    otg = Ruckig(robot.n_dof)
    trajectory = Trajectory(robot.n_dof)
 
    # Calculate the trajectory in an offline manner
    result = otg.calculate(inp, trajectory)
    if result == Result.ErrorInvalidInput:
        raise Exception('Invalid input!')
 
    print(f'Trajectory duration: {trajectory.duration:0.4f} [s]')

    '''start the ploting session'''
    sampling_interval = 0.01  # seconds
    time_points = np.linspace(0, trajectory.duration, 
                              num=int(trajectory.duration/sampling_interval)+1,
                              endpoint=True)

    positions = []
    velocities = []
    accelerations = []

    for t in time_points:
        pos, vel, acc = trajectory.at_time(t)
        positions.append(pos)
        velocities.append(vel)
        accelerations.append(acc)

    plt.figure(figsize=(10, 12))

    plt.subplot(3, 1, 1)
    for i in range(robot.n_dof):
        plt.plot(time_points, [p[i] for p in positions], label=f'DoF {i}')
    plt.ylabel('Position [m]')
    plt.title(f'Trajectory Profile (Sampling Interval: {sampling_interval}s)')
    plt.legend()
    # plt.grid(True)

    plt.subplot(3, 1, 2)
    for i in range(robot.n_dof):
        plt.plot(time_points, [v[i] for v in velocities], label=f'DoF {i}')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    # plt.grid(True)

    plt.subplot(3, 1, 3)
    for i in range(robot.n_dof):
        plt.plot(time_points, [a[i] for a in accelerations], label=f'DoF {i}')
    plt.ylabel('Acceleration [m/s²]')
    plt.xlabel('Time [s]')
    plt.legend()
    # plt.grid(True)

    plt.tight_layout()
    plt.show()

    robot.goto_given_conf(jnt_values=start_conf)
    robot.gen_meshmodel(rgb=[0,0,1], alpha=.3).attach_to(base)
    robot.goto_given_conf(jnt_values=goal_conf)
    robot.gen_meshmodel(rgb=[0,1,0], alpha=.3).attach_to(base)

    sampled_indices = np.linspace(0, len(positions)-1, num=10, dtype=int)
    sampled_positions = [positions[i] for i in sampled_indices]

    for conf in sampled_positions:
        robot.goto_given_conf(jnt_values=conf)
        robot.gen_meshmodel(alpha=.1).attach_to(base)
    
    base.run()