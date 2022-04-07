from locale import currency
import math
import pybullet

class Agent:
    
    def __init__(self, name, initial_pos, target_pos, color):
        self.name = name
        self.initial_pos = initial_pos
        self.current_pos = initial_pos
        self.target_pos = target_pos
        self.color = color
        self.obj = None
        
        
    def create(self):
        self.obj = pybullet.loadURDF('turtlebot3/turtlebot3_description/urdf/turtlebot3_waffle_pi.urdf.xacro', self.initial_pos, pybullet.getQuaternionFromEuler([0,0,0]))
        pybullet.changeVisualShape(self.obj, 0, rgbaColor=self.color)
        
        
    def getPos(self):
        self.current_pos = pybullet.getBasePositionAndOrientation(self.obj)
        return self.current_pos
    
    
    def step(self, action):
        #Action given includes quaternion
        angle = math.atan(action[1]/action[0])

        quarter = pybullet.getQuaternionFromEuler([0,0,angle])
        current_pos = self.getPos()
        pybullet.resetBasePositionAndOrientation(self.obj, [current_pos[0], current_pos[1], current_pos[2]], quarter)
        pybullet.resetBaseVelocity(self.obj, [action[0], action[1], 0])
        
        
    def stop(self):
        pybullet.resetBaseVelocity(self.obj, [0,0,0])
        
    
    def reward(self):
        self.current_pos = self.getPos()
        return math.dist((self.current_pos[0], self.current_pos[1], self.current_pos[2]), self.target_pos)
    
    def done(self):
        return self.reward() <= 0.1


    def reset(self):
        pybullet.resetBasePositionAndOrientation(self.obj, [self.initial_pos[0], self.initial_pos[1], self.initial_pos[2]])
        
