"""
Path Planning Sample Code with RRT*

author: Ahmed Qureshi, code adapted from AtsushiSakai(@Atsushi_twi)

"""


import argparse
import random
import math
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time


def diff(v1, v2):
    """
    Computes the difference v1 - v2, assuming v1 and v2 are both vectors
    """
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    """
    Computes the magnitude of the vector v.
    """
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    """
    Computes the Euclidean distance (L2 norm) between two points p1 and p2
    """
    return magnitude(diff(p1, p2))

class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea, alg, geom, dof=2, expandDis=0.05, goalSampleRate=5, maxIter=100):
        """
        Sets algorithm parameters

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,width,height],...]
        randArea:Ramdom Samping Area [min,max]

        """
        if geom == 'rectangle':
            new_start = [start[0], start[1], 0]
            self.start = Node(new_start)
            new_goal = [goal[0], goal[1], 0]
            self.end = Node(new_goal)
        else:
            self.start = Node(start)
            self.end = Node(goal)
        self.obstacleList = obstacleList
        self.minrandX = randArea[0]
        self.maxrandX = randArea[1]
        self.minrandY = randArea[2]
        self.maxrandY = randArea[3]
        self.alg = alg
        self.geom = geom
        self.dof = dof

        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter

        self.goalfound = False
        self.solutionSet = set()

    def planning(self, animation=False):
        """
        Implements the RTT (or RTT*) algorithm, following the pseudocode in the handout.
        You should read and understand this function, but you don't have to change any of its code - just implement the 3 helper functions.

        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        print(self.nodeList)
        for i in range(self.maxIter):
            rnd = self.generatesample()
            print(rnd)
            nind = self.GetNearestListIndex(self.nodeList, rnd)

            rnd_valid, rnd_cost = self.steerTo(rnd, self.nodeList[nind])

            if (rnd_valid):
                newNode = copy.deepcopy(rnd)
                newNode.parent = nind
                newNode.cost = rnd_cost + self.nodeList[nind].cost
                if self.alg == 'rrtstar':
                    nearinds = self.find_near_nodes(newNode) # you'll implement this method
                    newParent = self.choose_parent(newNode, nearinds) # you'll implement this method
                else:
                    newParent = None

                # insert newNode into the tree
                if newParent is not None:
                    newNode.parent = newParent
                    newNode.cost = dist(newNode.state, self.nodeList[newParent].state) + self.nodeList[newParent].cost
                else:
                    pass # nind is already set as newNode's parent
                self.nodeList.append(newNode)
                newNodeIndex = len(self.nodeList) - 1
                self.nodeList[newNode.parent].children.add(newNodeIndex)

                if self.alg == 'rrtstar':
                    self.rewire(newNode, newNodeIndex, nearinds) # you'll implement this method

                if self.is_near_goal(newNode):
                    self.solutionSet.add(newNodeIndex)
                    self.goalfound = True

                if animation:
                    self.draw_graph(rnd.state)

        return self.get_path_to_goal()

    def choose_parent(self, newNode, nearinds):
        """
        Selects the best parent for newNode. This should be the one that results in newNode having the lowest possible cost.

        newNode: the node to be inserted
        nearinds: a list of indices. Contains nodes that are close enough to newNode to be considered as a possible parent.

        Returns: index of the new parent selected
        """
        min_cost = newNode.cost
        min_index = None
        for index in nearinds:
            node = self.nodeList[index]
            valid, cost = self.steerTo(newNode, node)
            if valid and cost + node.cost < min_cost:
                min_index = index

        return min_index

    def steerTo(self, dest, source):
        """
        Charts a route from source to dest, and checks whether the route is collision-free.
        Discretizes the route into small steps, and checks for a collision at each step.

        This function is used in planning() to filter out invalid random samples. You may also find it useful
        for implementing the functions in question 1.

        dest: destination node
        source: source node

        returns: (success, cost) tuple
            - success is True if the route is collision free; False otherwise.
            - cost is the distance from source to dest, if the route is collision free; or None otherwise.
        """

        newNode = copy.deepcopy(source)

        DISCRETIZATION_STEP=self.expandDis

        dists = np.zeros(self.dof, dtype=np.float32)
        for j in range(0,self.dof):
            dists[j] = dest.state[j] - source.state[j]

        distTotal = magnitude(dists)


        if distTotal>0:
            incrementTotal = distTotal/DISCRETIZATION_STEP
            for j in range(0,self.dof):
                dists[j] =dists[j]/incrementTotal

            numSegments = int(math.floor(incrementTotal))+1

            stateCurr = None
            if self.geom == 'rectangle':
                stateCurr = np.zeros(self.dof + 1, dtype=np.float32)
                print(newNode.state)
                print(stateCurr)
                stateCurr[self.dof] = newNode.state[self.dof]
            else:
                stateCurr = np.zeros(self.dof,dtype=np.float32)
            for j in range(0,self.dof):
                stateCurr[j] = newNode.state[j]

            stateCurr = Node(stateCurr)

            for i in range(0,numSegments):

                if not self.__CollisionCheck(stateCurr):
                    return (False, None)

                for j in range(0,self.dof):
                    stateCurr.state[j] += dists[j]

            if not self.__CollisionCheck(dest):
                return (False, None)

            return (True, distTotal)
        else:
            return (False, None)

    def generatesample(self):
        """
        Randomly generates a sample, to be used as a new node.
        This sample may be invalid - if so, call generatesample() again.

        You will need to modify this function for question 3 (if self.geom == 'rectangle')

        returns: random c-space vector
        """
        if random.randint(0, 100) > self.goalSampleRate:
            sample=[]
            #for j in range(0,self.dof):
            sample.append(random.uniform(self.minrandX, self.maxrandX))
            sample.append(random.uniform(self.minrandY, self.maxrandY))
            if self.geom == 'rectangle':
                sample.append(random.uniform(0, 2*math.pi))
            rnd=Node(sample)
        else:
            rnd = self.end
        return rnd

    def is_near_goal(self, node):
        """
        node: the location to check

        Returns: True if node is within 5 units of the goal state; False otherwise
        """
        current_state = node.state
        current_end_state = self.end.state
        if self.geom == 'rectangle':
            current_state = [node.state[0], node.state[1]]
            current_end_state = [self.end.state[0], self.end.state[1]]
        d = dist(current_state, current_end_state)
        if d < 0.00005:
            return True
        return False

    @staticmethod
    def get_path_len(path):
        """
        path: a list of coordinates

        Returns: total length of the path
        """
        pathLen = 0
        for i in range(1, len(path)):
            pathLen += dist(path[i], path[i-1])

        return pathLen


    def gen_final_course(self, goalind):
        """
        Traverses up the tree to find the path from start to goal

        goalind: index of the goal node

        Returns: a list of coordinates, representing the path backwards. Traverse this list in reverse order to follow the path from start to end
        """
        path = [self.end.state]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            path.append(node.state)
            goalind = node.parent
        path.append(self.start.state)
        return path

    def find_near_nodes(self, newNode):
        """
        Finds all nodes in the tree that are "near" newNode.
        See the assignment handout for the equation defining the cutoff point (what it means to be "near" newNode)

        newNode: the node to be inserted.

        Returns: a list of indices of nearby nodes.
        """
        # Use this value of gamma
        GAMMA = 50
        num_nodes = len(self.nodeList) + 1
        cutoff = GAMMA*(math.log(num_nodes)/num_nodes)**(1/self.dof)
        near_indices = []
        for i in range(len(self.nodeList)):
            if dist(self.nodeList[i].state, newNode.state) <= cutoff:
                near_indices.append(i)

        # your code here
        return near_indices

    def rewire(self, newNode, newNodeIndex, nearinds):
        """
        Should examine all nodes near newNode, and decide whether to "rewire" them to go through newNode.
        Recall that a node should be rewired if doing so would reduce its cost.

        newNode: the node that was just inserted
        newNodeIndex: the index of newNode
        nearinds: list of indices of nodes near newNode
        """
        # your code here
        for node_ind in nearinds:
            node = self.nodeList[node_ind]
            valid, cost = self.steerTo(node, newNode)
            if valid and cost + newNode.cost < node.cost:
                node.parent = newNodeIndex
                node.cost = newNode.cost + cost
                self.nodeList[node_ind] = node
                self.updateChildren(node)

    def updateChildren(self, node):
        if node.children == None:
            return
        cost = node.cost
        for i in range(len(self.nodeList)):
            if self.nodeList[i] in node.children:
                self.nodeList[i].cost = cost + dist(node.state, self.nodeList[i].state)
                self.updateChildren(self.nodeList[i])

    def GetNearestListIndex(self, nodeList, rnd):
        """
        Searches nodeList for the closest vertex to rnd

        nodeList: list of all nodes currently in the tree
        rnd: node to be added (not currently in the tree)

        Returns: index of nearest node
        """
        dlist = []
        for node in nodeList:
            dlist.append(dist(rnd.state, node.state))

        minind = dlist.index(min(dlist))

        return minind
        


    def __CollisionCheck(self, node):
        """
        Checks whether a given configuration is valid. (collides with obstacles)

        You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
        """
        s = np.zeros(2, dtype=np.float32)
        s[0] = node.state[0]
        s[1] = node.state[1]


        
        for (ox, oy, sizex,sizey) in self.obstacleList:
            obs=[ox+sizex/2.0,oy+sizey/2.0]
            obs_size=[sizex,sizey]
            cf = False
            if self.geom == 'rectangle':
                # Rectangle with x, y, L = 1.5, W = 3, and theta
                theta = node.state[2]
                W = 1.5
                L = 3
                diag = math.sqrt((W/2)**2 + (L/2)**2)
                phi = theta - math.tanh((W/2)/(L/2))
                uL = [s[0] - L/2*math.cos(theta) + W/2*math.sin(theta), s[1] + L/2*math.sin(theta) + W/2*math.cos(theta)]
                uR = [s[0] + L/2*math.cos(theta) - W/2*math.sin(theta), s[1] + L/2*math.sin(theta) + W/2*math.cos(theta)]
                lL = [s[0] - L/2*math.cos(theta) + W/2*math.sin(theta), s[1] - L/2*math.sin(theta) - W/2*math.cos(theta)]
                lR = [s[0] + L/2*math.cos(theta) - W/2*math.sin(theta), s[1] - L/2*math.sin(theta) - W/2*math.cos(theta)]


                ouL = [ox, oy + obs_size[1]]
                ouR = [ox + obs_size[0], oy + obs_size[1]]
                olL = [ox, oy]
                olR = [ox + obs_size[0], oy]

                
                left_x = min(uL[0], lL[0], uR[0], lR[0])
                right_x = max(uR[0], lR[0], uL[0], lL[0])

                top_y = max(uL[1], uR[1], lL[1], lR[1])
                bottom_y = min(lL[1], lR[1], uL[1], uR[1])


                # Intended to do Separate Axis Theorem, but couldn't figure out
                # how to project obstacle onto robot axis

                if (right_x < ox or ox + sizex < left_x):
                    cf = True
                elif (top_y < oy or oy + sizey < bottom_y):
                    cf = True
                #elif ()
                

                # Utilized the below bounding circle check instead
                # If the distance between the centers of robot and obstacle
                # is greater than the radii of bounding spheres formed by the diagonals
                # of both combined, they don't collide

                center_dist = dist((node.state[0], node.state[1]), (obs[0], obs[1]))
                obs_diag = math.sqrt((sizex/2)**2 + (sizey/2)**2)

                if center_dist > diag + obs_diag:
                    cf = True


            elif self.geom == 'circle':
                free_ax = 0
                for j in range(self.dof):
                    # Circle with a radius of 1
                    upper_limit = float(s[j] + 1)
                    lower_limit = float(s[j] - 1)
                    obs_upper = float(obs[j]) + float(obs_size[j])/2
                    obs_lower = float(obs[j]) - float(obs_size[j])/2
                    if (upper_limit < obs_lower or obs_upper < lower_limit):
                        cf = True
                        break
            else:
                for j in range(self.dof):
                    if abs(obs[j] - s[j])>obs_size[j]/2.0:
                        cf=True
                        break


            if cf == False:
                return False

        return True  # safe'''

    def get_path_to_goal(self):
        """
        Traverses the tree to chart a path between the start state and the goal state.
        There may be multiple paths already discovered - if so, this returns the shortest one

        Returns: a list of coordinates, representing the path backwards; if a path has been found; None otherwise
        """
        if self.goalfound:
            goalind = None
            mincost = float('inf')
            for idx in self.solutionSet:
                cost = self.nodeList[idx].cost + dist(self.nodeList[idx].state, self.end.state)
                if goalind is None or cost < mincost:
                    goalind = idx
                    mincost = cost
            return self.gen_final_course(goalind)
        else:
            return None

    def draw_graph(self, rnd=None):
        """
        Draws the state space, with the tree, obstacles, and shortest path (if found). Useful for visualization.

        You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
        """
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for (ox, oy, sizex, sizey) in self.obstacleList:
            rect = mpatches.Rectangle((ox, oy), sizex, sizey, fill=True, color="purple", linewidth=0.1)
            plt.gca().add_patch(rect)


        for node in self.nodeList:
            if self.geom == 'circle':
                circle = plt.Circle((node.state[0], node.state[1]), 1, color='b')
                plt.gca().add_patch(circle)
            elif self.geom == 'rectangle':
                L = 3
                W = 1.5
                theta = math.degrees(node.state[2])
                #lower_left = [node.state[0] - L/2*math.cos(theta) + W/2*math.sin(theta), node.state[1] - L/2*math.sin(theta) - W/2*math.cos(theta)]
                lower_left = [node.state[0] - L/2, node.state[1] - W/2]
                rect = mpatches.Rectangle((lower_left[0], lower_left[1]), 3, 1.5, fill=True, color="blue", linewidth=0.1)
                rect_transform = matplotlib.transforms.Affine2D().rotate_deg_around(node.state[0], node.state[1], theta) + plt.gca().transData
                rect.set_transform(rect_transform)
                plt.gca().add_patch(rect)
            if node.parent is not None:
                if node.state is not None:
                    plt.plot([node.state[0], self.nodeList[node.parent].state[0]], [
                        node.state[1], self.nodeList[node.parent].state[1]], "-g")


        if self.goalfound:
            path = self.get_path_to_goal()
            x = [p[0] for p in path]
            y = [p[1] for p in path]
            plt.plot(x, y, '-r')

        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")

        plt.plot(self.start.state[0], self.start.state[1], "xr")
        plt.plot(self.end.state[0], self.end.state[1], "xr")
        plt.axis("equal")
        plt.axis([-20, 20, -20, 20])
        plt.grid(True)
        plt.pause(0.01)



class Node():
    """
    RRT Node
    """

    def __init__(self,state):
        self.state =state
        self.cost = 0.0
        self.parent = None
        self.children = set()



def main():
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 1')
    parser.add_argument('-g', '--geom', default='point', choices=['point', 'circle', 'rectangle'], \
        help='the geometry of the robot. Choose from "point" (Question 1), "circle" (Question 2), or "rectangle" (Question 3). default: "point"')
    parser.add_argument('--alg', default='rrt', choices=['rrt', 'rrtstar'], \
        help='which path-finding algorithm to use. default: "rrt"')
    parser.add_argument('--iter', default=100, type=int, help='number of iterations to run')
    parser.add_argument('--blind', action='store_true', help='set to disable all graphs. Useful for running in a headless session')
    parser.add_argument('--fast', action='store_true', help='set to disable live animation. (the final results will still be shown in a graph). Useful for doing timing analysis')

    args = parser.parse_args()

    show_animation = not args.blind and not args.fast

    print("Starting planning algorithm '%s' with '%s' robot geometry"%(args.alg, args.geom))
    starttime = time.time()


    obstacleList = [
    (-15,0, 15.0, 5.0),
    (15,-10, 5.0, 10.0),
    (-10,8, 5.0, 15.0),
    (3,15, 10.0, 5.0),
    (-10,-10, 10.0, 5.0),
    (5,-5, 5.0, 5.0),
    ]

    start = [-10, -17]
    goal = [10, 10]
    dof=2

    rrt = RRT(start=start, goal=goal, randArea=[-20, 20], obstacleList=obstacleList, dof=dof, alg=args.alg, geom=args.geom, maxIter=args.iter)
    path = rrt.planning(animation=show_animation)

    endtime = time.time()

    if path is None:
        print("FAILED to find a path in %.2fsec"%(endtime - starttime))
    else:
        print("SUCCESS - found path of cost %.5f in %.2fsec"%(RRT.get_path_len(path), endtime - starttime))
    # Draw final path
    if not args.blind:
        rrt.draw_graph()
        plt.show()


if __name__ == '__main__':
    main()
