#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Basic CARLA client example."""

from __future__ import print_function

import argparse
import logging
import random
import time
import os
import numpy as np

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line
from carla.transform import Transform #, Translation, Rotation, Scale
from carla import image_converter
import cv2


def run_carla_client(args):
    # Here we will run 3 episodes with 300 frames each.
    number_of_episodes = 15
    frames_per_episode = 10030
    #              [0  , 1  , 2  , 3  , 4  , 5  , 6 , 7, 8  , 9  , 10, 11, 12, 13, 14]
    # vehicles_num = [60, 60, 70, 50, 60, 60, 80, 60, 60, 60, 50, 70, 60, 50, 50]
    vehicles_num = [60, 60, 70, 50, 60, 60, 80, 60, 60, 60, 50, 70, 60, 50, 50]


    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        for episode in range(0, number_of_episodes):
            # Start a new episode.

            if args.settings_filepath is None:

                # Create a CarlaSettings object. This object is a wrapper around
                # the CarlaSettings.ini file. Here we set the configuration we
                # want for the new episode.
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=False,
                    NumberOfVehicles= vehicles_num[episode],#random.choice([0, 20, 15, 20, 25, 21, 24, 18, 40, 35, 25, 30]), #25,
                    NumberOfPedestrians=50,
                    DisableTwoWheeledVehicles=False,
                    WeatherId= episode, #1, #random.choice([1, 3, 7, 8, 14]),
                    QualityLevel=args.quality_level)
                settings.randomize_seeds()

                # Now we want to add a couple of cameras to the player vehicle.
                # We will collect the images produced by these cameras every
                # frame.
                #### Cameras aligned across the y-axis
                #### Horizontally shifted in the following Range
                # [-1.62, -1.08, -0.54, 0.0, 0.54, 1.08, 1.62]
                # LEFT RGB CAMERA
                y_locs = [-1.62, -1.08, -0.54, 0.0, 0.54, 1.08, 1.62]
                x_locs = [1.3, 1.84, 2.38, 2.92, 3.46, 4.0, 4.54]
                horizontal_cameras = {}
                for i,y_position in enumerate(y_locs):
                    # COLOR
                    camera_rgb = Camera('HorizontalCamera{0}RGB'.format(i),
                                     PostProcessing='SceneFinal')
                    camera_rgb.set_image_size(800, 600)
                    camera_rgb.set_position(x_locs[3], y_position, 1.50)
                    horizontal_cameras['HorizontalCamera{0}RGB'.format(i)] = camera_rgb
                    settings.add_sensor(camera_rgb)
                    # DEPTH
                    camera_depth = Camera('HorizontalCamera{0}Depth'.format(i),
                                          PostProcessing='Depth')
                    camera_depth.set_image_size(800, 600)
                    camera_depth.set_position(x_locs[3], y_position, 1.50)
                    horizontal_cameras['HorizontalCamera{0}Depth'.format(i)] = camera_depth
                    settings.add_sensor(camera_depth)
                    # SEGMENTATION
                    camera_seg = Camera('HorizontalCamera{0}Seg'.format(i),
                                       PostProcessing='SemanticSegmentation')
                    camera_seg.set_image_size(800, 600)
                    camera_seg.set_position(x_locs[3], y_position, 1.50)
                    horizontal_cameras['HorizontalCamera{0}Seg'.format(i)] = camera_seg
                    settings.add_sensor(camera_seg)

                forward_cameras = {}
                # z_locs = [1.5, 2.04, 2.58, 3.12, 3.66, 4.2, 4.74]

                # Cameras moving in to the scene
                # the are moved across the x axis
                # camera_90_p_ls.set_position(0.27, 1.0, 1.50)
                for i,x_position in enumerate(x_locs):
                    # COLOR
                    camera_rgb = Camera('ForwardCamera{0}RGB'.format(i),
                                     PostProcessing='SceneFinal')
                    camera_rgb.set_image_size(800, 600)
                    camera_rgb.set_position(x_position, y_locs[3], 1.5)
                    forward_cameras['ForwardCamera{0}RGB'.format(i)] = camera_rgb
                    settings.add_sensor(camera_rgb)
                    # DEPTH
                    camera_depth = Camera('ForwardCamera{0}Depth'.format(i),
                                          PostProcessing='Depth')
                    camera_depth.set_image_size(800, 600)
                    camera_depth.set_position(x_position, y_locs[3], 1.5)
                    forward_cameras['ForwardCamera{0}Depth'.format(i)] = camera_depth
                    settings.add_sensor(camera_depth)
                    # SEGMENTATION
                    camera_seg = Camera('ForwardCamera{0}Seg'.format(i),
                                       PostProcessing='SemanticSegmentation')
                    camera_seg.set_image_size(800, 600)
                    camera_seg.set_position(x_position, y_locs[3], 1.5)
                    forward_cameras['ForwardCamera{0}Seg'.format(i)] = camera_seg
                    settings.add_sensor(camera_seg)
            else:
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()
            scene = client.load_settings(settings)
            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))
            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode...')
            client.start_episode(player_start)
            horizontal_cameras_to_car = []
            for i in range(7):
                horizontal_cameras_to_car.append(
                    horizontal_cameras['HorizontalCamera{0}RGB'.format(i)].get_unreal_transform())
            forward_cameras_to_car = []
            for i in range(7):
                forward_cameras_to_car.append(
                    forward_cameras['ForwardCamera{0}RGB'.format(i)].get_unreal_transform())
            #
            # camera_90_p_l_to_car_transform = camera_90_p_l.get_unreal_transform()
            # camera_90_p_r_to_car_transform = camera_90_p_r.get_unreal_transform()

            # Create a folder for saving episode data
            if not os.path.isdir("/data/teddy/Datasets/carla_cross/Town01/episode_{:0>5d}".format(episode)):
                os.makedirs("/data/teddy/Datasets/carla_cross/Town01/episode_{:0>5d}".format(episode))

            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                # player_measurements = measurements.player_measurements
                world_transform = Transform(measurements.player_measurements.transform)

                # Compute the final transformation matrix.
                horizontal_cameras_to_world = []
                forward_cameras_to_world = []
                for i in range(7):
                    # print(type(world_transform))
                    # print(horizontal_cameras_to_car[i].shape)
                    # exit()
                    horizontal_cameras_to_world.append(world_transform * horizontal_cameras_to_car[i])
                    forward_cameras_to_world.append(world_transform * forward_cameras_to_car[i])
                # Save the images to disk if requested.
                if frame >= 30 and (frame % 2 == 0):
                    if args.save_images_to_disk:
                        for name, measurement in sensor_data.items():
                            filename = args.out_filename_format.format(episode, name, (frame-30)/2)
                            measurement.save_to_disk(filename)

                        # Save Transform matrix of each camera to separated files
                        for cam_num in range(7):
                            line = ""
                            filename = "{}episode_{:0>5d}/HorizontalCamera{}".format(args.root_path, episode, cam_num) + ".txt"
                            with open(filename, 'a+') as myfile:
                                for x in np.asarray(horizontal_cameras_to_world[cam_num].matrix[:3, :]).reshape(-1):
                                    line += "{:.8e} ".format(x)
                                line = line[:-1]
                                line += "\n"
                                myfile.write(line)
                                line = ""
                        # Forward Cameras
                        for cam_num in range(7):
                            line = ""
                            filename = "{}episode_{:0>5d}/ForwardCamera{}".format(args.root_path, episode, cam_num) + ".txt"
                            with open(filename, 'a+') as myfile:
                                for x in np.asarray(forward_cameras_to_world[cam_num].matrix[:3, :]).reshape(-1):
                                    line += "{:.8e} ".format(x)
                                line = line[:-1]
                                line += "\n"
                                myfile.write(line)
                                line = ""
                if not args.autopilot:
                    client.send_control(
                        steer=random.uniform(-1.0, 1.0),
                        throttle=0.5,
                        brake=0.0,
                        hand_brake=False,
                        reverse=False)
                else:
                    # Together with the measurements, the server has sent the
                    # control that the in-game autopilot would do this frame. We
                    # can enable autopilot by sending back this control to the
                    # server. We can modify it if wanted, here for instance we
                    # will add some noise to the steer.
                    control = measurements.player_measurements.autopilot_control
                    #control.steer += random.uniform(-0.1, 0.1)
                    client.send_control(control)
                #time.sleep(1)
            #myfile.close()
            #LeftCamera.close()
            #RightCamera.close()

def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '/data/teddy/Datasets/carla_cross/Town01/episode_{:0>5d}/{:s}/{:0>6d}'
    args.root_path = '/data/teddy/Datasets/carla_cross/Town01/'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
