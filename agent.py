import math

class Agent:
    
    def __init__(self, name, initial_pos, target_pos, color):
        self.name = name
        self.initial_pos = initial_pos
        self.target_pos = target_pos
        self.color = color
        self.obj = None
        
        
    def create(self):
        self.obj = pybullet.loadURDF('turtlebot3/turtlebot3_description/urdf/turtlebot3_waffle_pi.urdf.xacro', initial_pos, pybullet.getQuaternionFromEuler([0,0,0]))
        pybullet.changeVisualShape(self.obj, 0, rgbaColor=self.color)
        
        
    def getPos(self):
        return pybullet.getBasePositionAndOrientation(self.obj)
    
    
    def step(self, action):
        quarter = pybullet.getQuaternionFromEuler([0,0,action[2]])
        current_pos = self.getPos()
        pybullet.resetBasePositionAndOrientation(self.obj, [current_pos[0], current_pos[1], current_pos[2]], quarter)
        pybullet.resetBaseVelocity(self.obj, [action[0], action[1], 0])
        
        
    def stop(self):
        pybullet.resetBaseVelocity(self.obj, [0,0,0])
        
    
    def reward():
        current_pos = self.getPos()
        return math.dist((current_pos[0], current_pos[1], current_pos[2]), target_pos)
    
    

