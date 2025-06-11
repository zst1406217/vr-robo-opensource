# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the camera sensor from the Isaac Lab framework.

The camera sensor is created and interfaced through the Omniverse Replicator API. However, instead of using
the simulator or OpenGL convention for the camera, we use the robotics or ROS convention.

.. code-block:: bash

    # Usage with GUI
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/run_usd_camera.py --enable_cameras

    # Usage with headless
    ./isaaclab.sh -p source/standalone/tutorials/04_sensors/run_usd_camera.py --headless --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
import copy
import socket
import pickle
from multiprocessing import Process, Queue
import torch
import traceback

class TensorServer:
    def __init__(self, host='localhost', port=12345, time_out=10):
        self.host = host
        self.port = port
        self.time_out = time_out
        self.queue = Queue()
        self.tensor = torch.zeros([2, 3, 240, 320])
        
    def receive_tensor(self, host='localhost', port=12345):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        conn, addr = s.accept()
        conn.settimeout(self.time_out)  # Set timeout for receiving data
        data = b""
        try:
            while True:
                packet = conn.recv(40960000)
                if not packet:
                    break
                data += packet
        except socket.timeout:
            print("No new tensor received for 5 seconds, terminating connection.")
        finally:
            conn.close()
        tensor = torch.tensor(pickle.loads(data), device="cuda")
        return tensor

    def start(self):
        self.process = Process(target=self.run)
        self.process.start()
        
    def close(self):
        self.process.terminate()

    def run(self):
        while True:
            tensor = self.receive_tensor(self.host, self.port)
            self.queue.put(tensor)

    def get_tensor(self):
        if not self.queue.empty():
            self.tensor = self.queue.get()
            return self.tensor
        return self.tensor

# Start the tensor server
tensor_server = TensorServer()
tensor_server.start()
    
    
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import random
from PIL import Image

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.utils.math as math_utils
import omni.replicator.core as rep

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.utils import convert_dict_to_backend

@torch.jit.script
def form_from_world_to_ros(quat):
    quat_ = torch.tensor([0.5, -0.5, 0.5, -0.5], device=quat.device)
    quat_ros = torch.empty_like(quat)
    quat_ros[:, 0] = quat[:, 0] * quat_[0] - quat[:, 1] * quat_[1] - quat[:, 2] * quat_[2] - quat[:, 3] * quat_[3]
    quat_ros[:, 1] = quat[:, 0] * quat_[1] + quat[:, 1] * quat_[0] + quat[:, 2] * quat_[3] - quat[:, 3] * quat_[2]
    quat_ros[:, 2] = quat[:, 0] * quat_[2] - quat[:, 1] * quat_[3] + quat[:, 2] * quat_[0] + quat[:, 3] * quat_[1]
    quat_ros[:, 3] = quat[:, 0] * quat_[3] + quat[:, 1] * quat_[2] - quat[:, 2] * quat_[1] + quat[:, 3] * quat_[0]
    return quat_ros


def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    # In contrast to the ray-cast camera, we spawn the prim at these locations.
    # This means the camera sensor will be attached to these prims.
    prim_utils.create_prim("/World/Origin_00", "Xform")
    prim_utils.create_prim("/World/Origin_01", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        height=240,
        width=320,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, horizontal_aperture=28.631485, vertical_aperture=21.468225, clipping_range=(0.01, 100.0)
        ),
    )
    # Calculate field of view (FOV) in x and y directions
    fovx = 2 * np.arctan(camera_cfg.spawn.horizontal_aperture / (2 * camera_cfg.spawn.focal_length))
    fovy = 2 * np.arctan(camera_cfg.spawn.vertical_aperture / (2 * camera_cfg.spawn.focal_length))
    print(f"FOVx: {fovx} radians, FOVy: {fovy} radians")
    # Create camera
    camera = Camera(cfg=camera_cfg)

    return camera


def design_scene() -> dict:
    """Design the scene."""
    # Populate scene
    # -- Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # -- Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create a dictionary for the scene entities
    scene_entities = {}

    # Xform to hold objects
    prim_utils.create_prim("/World/Objects", "Xform")
    
    cfg = sim_utils.UsdFileCfg(usd_path="./pgsr_obj_transformed.usd")
    cfg.func("/World/Objects/sqz_lab", cfg, translation=(0.0, 0.0, 0.0))

    # Sensors
    camera = define_sensor()

    # return the scene information
    scene_entities["camera"] = camera
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # extract entities for simplified notation
    camera: Camera = scene_entities["camera"]
    
    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    # Camera positions, targets, orientations
    camera_positions = torch.tensor([[-1.5, 0, 0.5], [-1.13681074, 0.37103293, 0.93501559]], device=sim.device)
    # These orientations are in ROS-convention, and will position the cameras to view the origin
    camera_orientations = torch.tensor(  # noqa: F841
        [[0.5, -0.5, 0.5, -0.5], [0.281, -0.410, 0.651, -0.574]], device=sim.device
    )

    # Set pose: There are two ways to set the pose of the camera.
    # -- Option-1: Set pose using view
    # camera.set_world_poses_from_view(camera_positions, camera_targets)
    # -- Option-2: Set pose using ROS
    camera.set_world_poses(camera_positions, camera_orientations, convention="ros")

    # Index of the camera to use for visualization and saving
    camera_index = args_cli.camera_id

    # Create the markers for the --draw option outside of is_running() loop
    if sim.has_gui() and args_cli.draw:
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)
        
    conn = rpyc.connect('localhost', 18861)
    run_count = 0

    # Simulate physics
    while simulation_app.is_running():
        run_count += 1
        # Step simulation
        sim.step()
        # Update camera data
        if run_count<50:
            camera_positions = torch.tensor([[-1.5, 0, 0.5], [-1.13681074, 0.37103293, 0.93501559]], device=sim.device)
            # These orientations are in ROS-convention, and will position the cameras to view the origin
            camera_orientations = torch.tensor(  # noqa: F841
                [[0.5, -0.5, 0.5, -0.5], [0.281, -0.410, 0.651, -0.574]], device=sim.device
            )
        else:
            camera_positions = torch.tensor([[-1.5, 0-(run_count-50)*0.01, 0.5], [-1.13681074, 0.37103293, 0.93501559]], device=sim.device)
            # These orientations are in ROS-convention, and will position the cameras to view the origin
            camera_orientations = torch.tensor(  # noqa: F841
                [[0.5, -0.5, 0.5, -0.5], [0.281, -0.410, 0.651, -0.574]], device=sim.device
            )
        
        camera.set_world_poses(camera_positions, camera_orientations, convention="ros")
        camera.update(dt=sim.get_physics_dt())
        
        pos_data = camera.data.pos_w
        ori_data = camera.data.quat_w_world
        ori_data = math_utils.convert_camera_frame_orientation_convention(ori_data, "world", "ros")

        conn.root.render(pos_data, ori_data)
        images = tensor_server.get_tensor()
        if images is not None:
            print(images.shape)
            ndarr = images[0].permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            # im = Image.fromarray(ndarr)
            # im.save('./scene_demo/test.png')
            sim_ndarr = camera.data.output["rgb"][0].cpu().numpy()
            # sim_im = Image.fromarray(sim_ndarr)
            # sim_im.save('./sim.png')
            concat_im = np.concatenate((ndarr, sim_ndarr), axis=1)
            concat_im = Image.fromarray(concat_im)
            concat_im.save('./scene_demo/concat'+str(run_count)+'.png')
        else:
            print("None")

        # Extract camera data
        if args_cli.save:
            # Save images from camera at camera_index
            # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )

            # Extract the other information
            single_cam_info = camera.data.info[camera_index]

            # Pack data back into replicator format to save them using its writer
            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            # Save images
            # Note: We need to provide On-time data for Replicator to save the images.
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)

        # Draw pointcloud if there is a GUI and --draw has been passed
        if sim.has_gui() and args_cli.draw and "distance_to_image_plane" in camera.data.output.keys():
            # Derive pointcloud from camera at camera_index
            pointcloud = create_pointcloud_from_depth(
                intrinsic_matrix=camera.data.intrinsic_matrices[camera_index],
                depth=camera.data.output[camera_index]["distance_to_image_plane"],
                position=camera.data.pos_w[camera_index],
                orientation=camera.data.quat_w_ros[camera_index],
                device=sim.device,
            )

            # In the first few steps, things are still being instanced and Camera.data
            # can be empty. If we attempt to visualize an empty pointcloud it will crash
            # the sim, so we check that the pointcloud is not empty.
            if pointcloud.size()[0] > 0:
                pc_markers.visualize(translations=pointcloud)

def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # design the scene
    scene_entities = design_scene()
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run simulator
    run_simulator(sim, scene_entities)

if __name__ == "__main__":
    # run the main function
    try:
        main()
        simulation_app.close()
    except Exception as e:
        traceback.print_exc()
        tensor_server.close()
    # close sim app
    tensor_server.close()
