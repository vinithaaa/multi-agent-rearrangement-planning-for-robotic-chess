import gym 
from gym import spaces
import numpy as np
from agent import Agent
import pybullet
import pybullet_data
import time
import math

class Environment():

    def __init__(self, collisionPenalty, actionDuration):
        self.collisionPenalty = collisionPenalty
        self.actionDuration = actionDuration

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
        
        # initialize action space and agent
        width, height, rgbImg, depthImg, segImg = pybullet.getCameraImage(width = 224, height = 224, viewMatrix=viewMatrix, projectionMatrix = projectionMatrix)
        self.agents = self.createAgents()
        self.n = len(self.agents)
        #actionBox = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        actionBox = spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)
        #obsBox = spaces.Box(low=np.array([0.0,-0.5]), high=np.array([2.0,1.5]), dtype=np.float32)
        obsBox = spaces.Box(low=np.array([-2.0,-2.0]), high=np.array([3.0,3.0]), dtype=np.float32)
        self.action_space = [actionBox] * self.n
        self.observation_space = [obsBox] * self.n
        pybullet.setRealTimeSimulation(1)

    # create agents for each scenario 
    def createAgents(self):
        #first = Agent('Start', (0.25, 0.25, 0), (3.25, 0.25, 0), [0,0,1,1])
        first = Agent('Start', (0.25, 0.25, 0), (2.25, 0.25, 0), [0,0,1,1])
        obs1 = Agent('Obs1', (0.75, 0.25, 0), (0.75, 0.25, 0), [0,1,0,1])
        obs11 = Agent('Obs11', (0.75, -0.25, 0), (0.75, -0.25, 0), [0,1,0,1])
        obs12 = Agent('Obs12', (0.75, 0.75, 0), (0.75, 0.75, 0), [0,1,0,1])
        #obs13 = Agent('Obs13', (0.75, -0.75, 0), (0.75, -0.25, 0), [0,1,0,1])
        #obs14 = Agent('Obs14', (0.75, 1.25, 0), (0.75, 1.25, 0), [0,1,0,1])
        obs2 = Agent('Obs2', (1.25, 0.25, 0), (1.25, 0.25, 0), [0,1,0,1])
        obs21 = Agent('Obs21', (1.25, -0.25, 0), (1.25, -0.25, 0), [0,1,0,1])
        obs22 = Agent('Obs22', (1.25, 0.75, 0), (1.25, 0.75, 0), [0,1,0,1])
        #obs23 = Agent('Obs23', (1.25, -0.75, 0), (1.25, -0.25, 0), [0,1,0,1])
        #obs24 = Agent('Obs24', (1.25, 1.25, 0), (1.25, 1.25, 0), [0,1,0,1])
        obs3 = Agent('Obs3', (1.75, 0.25, 0), (1.75, 0.25, 0), [0,1,0,1])
        obs31 = Agent('Obs31', (1.75, -0.25, 0), (1.75, -0.25, 0), [0,1,0,1])
        obs32 = Agent('Obs32', (1.75, 0.75, 0), (1.75, 0.75, 0), [0,1,0,1])
        #obs4 = Agent('Obs4', (2.25, 0.25, 0), (2.25, 0.25, 0), [0,1,0,1])
        #obs41 = Agent('Obs41', (2.25, -0.25, 0), (2.25, -0.25, 0), [0,1,0,1])
        #obs42 = Agent('Obs42', (2.25, 0.75, 0), (2.25, 0.75, 0), [0,1,0,1])
        #obs5 = Agent('Obs5', (2.75, 0.25, 0), (2.75, 0.25, 0), [0,1,0,1])
        #obs51 = Agent('Obs51', (2.75, -0.25, 0), (2.75, -0.25, 0), [0,1,0,1])
        #obs52 = Agent('Obs52', (2.75, 0.75, 0), (2.75, 0.75, 0), [0,1,0,1])
        return [first, obs1, obs11, obs12, obs2, obs21, obs22, obs3, obs31, obs32]
        #first = Agent('First', (2.25, 1.25, 0), (0.25, 0.25, 0), [0,0,1,1])
        #second = Agent('Second', (2.25, 0.25, 0), (0.25, 1.25, 0), [0,1,0,1])
        #third = Agent('Third', (2.25, 0.75, 0), (0.25, 0.75, 0), [1,0,0,1])
        #return [first, second, third]
        """
        goal = (1.75, 0.75, 0)
        start = (0.25, 0.25, 0)
        player = Agent('Player', start, goal, [0,0,1,1])
        obs1 = Agent('Obs1', [0.75,0.25,0], [0.75,0.25,0], [0,1,0,1])
        obs2 = Agent('Obs2', [0.75,0.75,0], [0.75,0.75,0], [0,1,0,1])
        obs3 = Agent('Obs3', [0.75,-0.25,0], [0.75,-0.25,0], [0,1,0,1])
        obs4 = Agent('Obs4', [1.25,0.25,0],[1.25,0.25,0],[0,1,0,1])
        obs5 = Agent('Obs5', [1.25,0.75,0], [1.25,0.75,0],[0,1,0,1])
        obs6 = Agent('Obs6', [1.25,-0.25,0], [1.25,-0.25,0],[0,1,0,1])
        #target = Agent('Target', goal, [1.75,1,0], [1,0,0,1])
        target = Agent('Target', goal, start, [1,0,0,1])
        return [player, obs1, obs2, obs3, obs4, obs5, obs6, target]
        """

    # check to see if all of the agents are done with their task (getting to the target location)
    def doneGlobal(self):
        for agent in self.agents:
            if agent.done() == False:
                return False
        return True


    # have each agent take an action        
    def step(self, actions):
        for x in range(self.actionDuration):
            pybullet.stepSimulation()
            time.sleep(1./240.)
            for i, agent in enumerate(self.agents):
                agent.step(actions[i])

        pybullet.stepSimulation()
        time.sleep(1./240.)
        for i, agent in enumerate(self.agents):
            agent.stop()

        observations = []
        rewards = []
        done = []
        # calculate rewards for each agent 
        for agent in self.agents:
            done.append(agent.done())
            observations.append(self.observe(agent))
            agentReward = agent.reward()
            for other in self.agents:
                if agent != other and self.collision(agent, other):
                     agentReward -= self.collisionPenalty  
            rewards.append(agentReward)
        
        #done = [self.doneGlobal()] * len(rewards)

        print("Step Rewards: ", rewards)
        return observations, rewards, done

    # check to see if there's a collision between two agents
    def collision(self, firstAgent, secondAgent):
        firstXY = (firstAgent.getPos()[0][0], firstAgent.getPos()[0][1])
        secondXY = (secondAgent.getPos()[0][0], secondAgent.getPos()[0][1])
        return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(firstXY, secondXY))) <= 0.25
        #return math.dist([firstAgent.getPos()[0], firstAgent.getPos()[1]],[secondAgent.getPos[0], secondAgent.getPos[1]]) <= 0.25
        

    def observe(self, agent):
        position = agent.getPos()[0]
        posNP = np.array([[position[0]], [position[1]]])
        return np.concatenate(posNP)


    # reset environment
    def reset(self):
        obs = []
        for agent in self.agents:
            agent.reset()
            obs.append(self.observe(agent))
        return obs
        #obs = []
        #pybullet.resetSimulation()
        #self.agents = self.createAgents()
        #for agent in self.agents:
        #    obs.append(self.observe(agent))
        #return obs
    
    

