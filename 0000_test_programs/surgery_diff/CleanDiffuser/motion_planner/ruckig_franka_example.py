from copy import copy

import wrs.robot_sim.manipulators.franka_research_3_arm.franka_research_3_arm as franka
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
 
from ruckig import InputParameter, OutputParameter, Result, Ruckig

def initialize():
    '''init the robot and world'''
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = franka.FrankaResearch3Arm(enable_cc=True)
    inp = InputParameter(robot.n_dof)
    out = OutputParameter(robot.n_dof)
    otg = Ruckig(robot.n_dof, 0.001)
    
    return base, robot, otg, inp, out

if __name__ == '__main__':
    # Initialize the world and robot
    base, robot, otg, inp, out = initialize()
 
    # Generate random start and goal configurations
    start_conf = robot.rand_conf()
    goal_conf = robot.rand_conf()
    print('start configuration:', repr(start_conf))
    print('goal configuration:', repr(goal_conf))

    inp.current_position, inp.target_position = start_conf, goal_conf
    inp.current_velocity = rm.np.zeros(robot.n_dof) # support non-zero vel, zero for simplicity
    inp.current_acceleration = rm.np.zeros(robot.n_dof) # support non-zero acc, zero for simplicity
    inp.target_velocity = rm.np.zeros(robot.n_dof)
    inp.target_acceleration = rm.np.zeros(robot.n_dof)
    inp.min_position = robot.jnt_ranges[:, 0]
    inp.max_position = robot.jnt_ranges[:, 1]

    inp.max_velocity = rm.np.asarray([rm.pi * 2 / 3] * robot.n_dof)
    inp.max_acceleration = rm.np.asarray([rm.pi] * robot.n_dof)
    inp.max_jerk = rm.np.asarray([rm.pi * 2] * robot.n_dof)
 
    print('\t'.join(['t'] + [str(i) for i in range(otg.degrees_of_freedom)]))
 
    # Generate the trajectory within the control loop
    first_output, out_list = None, []
    res = Result.Working
    while res == Result.Working:
        res = otg.update(inp, out)
 
        print('\t'.join([f'{out.time:0.3f}'] + [f'{p:0.3f}' for p in out.new_position]))
        out_list.append(copy(out))
 
        out.pass_to_input(inp)
 
        if not first_output:
            first_output = copy(out)
 
    print(f'Calculation duration: {first_output.calculation_duration:0.1f} [µs]')
    print(f'Trajectory duration: {first_output.trajectory.duration:0.4f} [s]')
 
    # Plot the trajectory
    from pathlib import Path
    from plotter import Plotter
    import os
 
    project_path = Path(__file__).parent.parent.absolute()
    Plotter.plot_trajectory(os.path.join(project_path, '01_trajectory.pdf'), otg, inp, out_list, plot_jerk=True)