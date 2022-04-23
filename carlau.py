import time
import random
import atexit
import config
import numpy as np
import cv2
import carla

class Carla:
    def __init__(self, autopilot, spawn_point):
        self.frame = None
        self.vehicle = None
        self.autopilot = autopilot
        self.spawn_point = spawn_point
        self.env_init()

    def env_init(self):
        client = carla.Client('localhost', 2000)
        world = client.get_world()
        try:
            world = client.load_world('Town05') 
        except RuntimeError:
            print('loading carla world')
            client.set_timeout(35)
    
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter('model3')[0]
        if self.spawn_point == False:
            spawn_point = random.choice(world.get_map().get_spawn_points())
        else:
            spawn_point = world.get_map().get_spawn_points()[self.spawn_point]
        self.vehicle = world.spawn_actor(bp, spawn_point)
        self.vehicle.set_autopilot(self.autopilot)

        blueprint = blueprint_library.find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', f'{1280}')
        blueprint.set_attribute('image_size_y', f'{720}')
        blueprint.set_attribute('fov', '70')
        transform = carla.Transform(carla.Location(x=0.4, z=1.2))
        
        sensor = world.spawn_actor(blueprint, transform, attach_to=self.vehicle)
        sensor.listen(self._frame_processing)

        def destroy():
            sensor.destroy()
        atexit.register(destroy)

    def _frame_processing(self, image):
        img = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        img = img.reshape(720, 1280, 4)
        self.frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

if __name__ == '__main__':
    Carla(False, True)