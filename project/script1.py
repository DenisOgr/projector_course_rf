import scipy.misc as smp
import gym
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

env = gym.make('Assault-v0')


def get_player_coords(observation):
    observation = np.asarray(smp.toimage(observation).convert('L'))
    for pixel in range(0, observation.shape[1], 1):
        color_pixel = observation[219, pixel]
        if color_pixel != 0:
            last_color_of_player = observation[219, pixel + 7]
            next_color_of_player = observation[219, pixel + 8]
            if (last_color_of_player != 0 & next_color_of_player == 0):
                return 215, pixel + 4


def get_ships_coords(observation):
    observation_start = observation[80:210:, :,:]

    observation_2d = np.asarray(smp.toimage(observation_start[:,:,0]).convert('L'))
    observation_2d_non_zero_rc = np.nonzero(observation_2d)
    observation_2d_non_zero = np.empty((0, 2), int)
    for k in range(len(observation_2d_non_zero_rc[0])):
        row = np.array([[observation_2d_non_zero_rc[0][k], observation_2d_non_zero_rc[1][k]]])
        observation_2d_non_zero = np.append(observation_2d_non_zero, row, axis=0)

    if observation_2d_non_zero.size == 0:
        return observation_2d_non_zero
    bandwidth = estimate_bandwidth(observation_2d_non_zero, quantile=0.3)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(observation_2d_non_zero)

    return (ms.cluster_centers_ + [80, 0]).astype(int)

def get_centre_of_ships(ships_coord):
    result = np.mean(ships_coord, axis=0).astype(int)
    return result[0], result[1]

def print_debbug_with_image(observation, player_coords, ships_coord, centr_of_ships):
    print('Player centre color: RED(255, 0,0)')
    print('Ships centrs color: GREEN(0, 255, 0)')
    print('Centre of all ships: BLUE(0,0, 255)')
    observation[player_coords] = [255,0,0]
    for center in ships_coord:
        observation[center[0], center[1]] = [0,255,0]
    observation[centr_of_ships[0], centr_of_ships[1]] = [0,0,255]
    img = smp.toimage(observation)
    img.show()

def print_debbug(player_coords, ships_coord, centre_of_ships):
    print('Player coords: ' , player_coords)
    print('Ships coords: ' , ships_coord)
    print('Centre of all ships coords: ' , centre_of_ships)


number_tic_for_test = 25
for i_episode in range(20):
    print("Episode#", i_episode)
    observation = env.reset()
    for t in range(100):
        env.render()
        player_y, player_x = get_player_coords(observation)
        ships_coord = get_ships_coords(observation)
        if ships_coord.size != 0:
            centr_of_ships_y, centr_of_ships_x = get_centre_of_ships(ships_coord)
            if number_tic_for_test == t:
                print_debbug_with_image(observation, (player_y, player_x), ships_coord, (centr_of_ships_y, centr_of_ships_x))

            print_debbug((player_y, player_x), ships_coord, (centr_of_ships_y, centr_of_ships_x))
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    print('Stop')
