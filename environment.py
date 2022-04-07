import gym 
from gym import spaces
import numpy as np
from agent import Agent
import pybullet
from pybullet import pybullet_data
import time
import math

class Environment():

    def __init__(self):
        self.action_space = spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0.0,-0.5]), high=np.array([2.0,1.5]), dtype=np.float32)
        self.agents = self.createAgents()
        # Initializing pybullet
        physicsClient = pybullet.connect(pybullet.GUI)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0,0,-9.8)

        # Loading chessgrid plane
        planeID = pybullet.loadURDF('plane.urdf')
        chess_grid = pybullet.loadTexture('checker_huge.gif')
        pybullet.changeVisualShape(planeID, -1, textureUniqueId=chess_grid)

        # Setting up overhead camera
        viewMatrix = pybullet.computeViewMatrix(
            cameraEyePosition=[0.75,0.25,3],
            cameraTargetPosition=[0.75,0.25,0],
            cameraUpVector=[0,1,0])

        projectionMatrix = pybullet.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1)

        width, height, rgbImg, depthImg, segImg = pybullet.getCameraImage(width = 224, height = 224, viewMatrix=viewMatrix, projectionMatrix = projectionMatrix)

        pybullet.setRealTimeSimulation(1)

    def createAgents(self):
        goal = (1.25, 0.75, 0)
        start = (0.25, 0.25, 0)
        player = Agent('Player', start, goal, [0,0,1,1])
        obs1 = Agent('Obs1', [0.75,0.25,0], [0.75,0.25,0], [0,1,0,1])
        obs2 = Agent('Obs2', [0.75,0.75,0], [0.75,0.75,0], [0,1,0,1])
        obs3 = Agent('Obs3', [0.75,-0.25,0], [0.75,-0.25,0], [0,1,0,1])
        target = Agent('Target', goal, [1.75,1,0], [1,0,0,1])
        return [player, obs1, obs2, obs3, target]
    

    def done(self):
        for agent in self.agents:
            if agent.done() == False:
                return False
        return True


            
    def step(self, actions):
        for x in range(10):
            pybullet.stepSimulation()
            time.sleep(1./240.)
            for i, agent in enumerate(self.agents):
                agent.step(actions[i])

        for i, agent in enumerate(self.agents):
            agent.stop()


        observations = []
        rewards = []
        for agent in self.agents:
            observations.append(agent.getPos())
            agentReward = agent.reward()
            for other in self.agents:
                if agent != other and self.collision(agent, other):
                     agentReward -= 1  
            rewards.append(agentReward)
        
        done = [self.done()] * len(rewards)

        return observations, rewards, done

    def collision(firstAgent, secondAgent):
        return math.dist([firstAgent.getPos()[0], firstAgent.getPos()[1]],[secondAgent.getPos[0], secondAgent.getPos[1]]) <= 0.25
        
    def reset(self):
        for agent in self.agents:
            agent.reset()
    
    

