import time
import random
import atexit
import torch
import numpy as np
import cv2
import carla
from model import Model

W, H = 1600, 900
frame = None

model = Model()
model_state = torch.load('/autopilot-carla/model.pth')
model.load_state_dict(model_state)
model.eval()

def frame_processing(image):
    global frame
    img = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    img = img.reshape(H, W, 4)
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
   
def main():
    run = True
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)
    
    # load map
    world = client.get_world()
    world = client.load_world('Town05') 
    
    # spawn car
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)

    vehicle.set_autopilot(False)

    # attach camera to car
    blueprint = blueprint_library.find('sensor.camera.rgb')
    blueprint.set_attribute('image_size_x', f'{W}')
    blueprint.set_attribute('image_size_y', f'{H}')
    blueprint.set_attribute('fov', '70')
    transform = carla.Transform(carla.Location(x=0.4, z=1.2))
    sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
    # get frame from camera
    sensor.listen(frame_processing)

    def destroy():
        sensor.destroy()
    atexit.register(destroy)

    # main loop
    while run:
        if frame is not None:    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                run = False

            # ignore traffic lights     
            if vehicle.is_at_traffic_light():
                traffic_light = vehicle.get_traffic_light()
                traffic_light.set_state(carla.TrafficLightState.Green)
           
            # show frame
            cv2.imshow('', frame)
            cv2.waitKey(1)
            
            # fps delay
            time.sleep(0.060) 
        
            # use model to control car
            data = cv2.resize(frame, (200, 66))
            data = torch.tensor(data).float()
            with torch.no_grad():
                out = model(data.reshape(1, 3, 66, 200))
                vehicle.apply_control(carla.VehicleControl(throttle=float(out[0][0]), steer=float(out[0][1])))
            
if __name__ == '__main__':
    main()