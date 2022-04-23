import time
import random
import atexit
import config
import numpy as np
import cv2
import carla
import matplotlib.pyplot as plt
from carlau import Carla

class DataGenerator(Carla):
    def __init__(self):
        super().__init__(True, False)
        self.x_data = [] 
        self.y_data = []
        self.main()
    
    def main(self):
        run = True
        count = 0
        vehicle = self.vehicle
        dataset_size = config.TEST_DATASET_SIZE+config.TRAIN_DATASET_SIZES

        while run:
            if self.frame is not None:    
                if vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    traffic_light.set_state(carla.TrafficLightState.Green)
                    
                cv2.imshow('', cv2.resize(self.frame, (config.WIDTH, config.HEIGHT)))
                cv2.waitKey(50)
                vehicle_data = vehicle.get_control()
                steer = vehicle_data.steer
                self._collect_data(steer)
                count += 1

                if count >= dataset_size:
                    run = False
                
                print(f'[{count}/{dataset_size}]')
                
        np.save(config.DATASET_PATH+'y_train.npy', np.array(self.y_data), allow_pickle=True)
        np.save(config.DATASET_PATH+'x_train.npy', np.array(self.x_data), allow_pickle=True)
    
    def _collect_data(self, steer):
        frame = self.frame
        frame = cv2.resize(frame, (200, 66))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.x_data.append(frame)
        self.y_data.append([steer])
                    
if __name__ == '__main__':
    DataGenerator()