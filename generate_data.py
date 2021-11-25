import time
import random
import atexit
import numpy as np
import cv2
import carla

W, H = 1600, 900
frame = None

x_data = []
y_data = []

def collect_data(frame, count, steer, throttle):
    frame = cv2.resize(frame, (200, 66))
    x_data.append(frame)
    y_data.append([throttle, steer])

def frame_processing(image):
    global frame
    img = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    img = img.reshape(H, W, 4)
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
   
def main():
    count = 0
    run = True
    new_frame_time = 0
    prev_frame_time = 0

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

    vehicle.set_autopilot(True)

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
        
            # fps counter
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
                
            
            cv2.putText(frame, f'data generated:{count}', (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f'fps:{int(fps)}', (7, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, 'press q to finish', (7, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)
           
            # show frame
            cv2.imshow('', frame)
            cv2.waitKey(1)
            
            # fps delay
            time.sleep(0.060) 
        
            # get car data  
            vehicledata = vehicle.get_control()
            throttle = vehicledata.throttle
            steer = vehicledata.steer
            
            # add  data to list
            collect_data(frame, count, steer, throttle)
            count += 1
                  
    np.save('/autopilot-carla/y_train.npy', np.array(y_data), allow_pickle=True)
    np.save('/autopilot-carla/x_train.npy', np.array(x_data), allow_pickle=True)

if __name__ == '__main__':
    main()
    