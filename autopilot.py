import random
import atexit
import config
import torch
import numpy as np
import cv2
import carla
from model import Model
from carlau import Carla

class Autopilot(Carla):
    def __init__(self, model, model_state):
        super().__init__(False, 10)
        self.model = model
        self.model_state = model_state
        self.main()
    
    def main(self):
        run = True
        model = self.model
        vehicle = self.vehicle
        model_state = torch.load(self.model_state)
        model.load_state_dict(model_state)
        model.eval()
        
        while run:
            if self.frame is not None:    
                if vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    traffic_light.set_state(carla.TrafficLightState.Green)
                    
                cv2.imshow('"q" - exit', cv2.resize(self.frame, (config.WIDTH, config.HEIGHT)))
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                self._control_vehicle(model, vehicle)
    
    def _control_vehicle(self, model, vehicle):
        data = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        data = cv2.resize(data, (200, 66))
        data = torch.tensor(data).reshape(1, 3, 66, 200).float()
        with torch.no_grad():
            out = model(data).reshape(1)
            vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=float(out[0])))

if __name__ == '__main__':
    Autopilot(Model(), 'model_state.pth')   