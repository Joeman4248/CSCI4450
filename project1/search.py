# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


class Node:
    def __init__(self, state: tuple, path=[], cost=0):
        self.state = state
        self.path = path
        self.cost = cost


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def graphSearch(problem: SearchProblem, fringe):
    """ Generic graph search algorithm """
    explored = set()

    fringe.push(Node(problem.getStartState()))

    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoalState(node.state): return node.path
        if node.state in explored: continue
        explored.add(node.state)
        for (state, action, _) in problem.getSuccessors(node.state):
            child = Node(state, node.path + [action])
            fringe.push(child)


def depthFirstSearch(problem: SearchProblem):
    """ Search the deepest nodes in the search tree first. """
    return graphSearch(problem, util.Stack())


def breadthFirstSearch(problem: SearchProblem):
    """ Search the shallowest nodes in the search tree first. """
    return graphSearch(problem, util.Queue())


def uniformCostSearch(problem: SearchProblem):
    """ Search the node of least total cost first. """
    return aStarSearch(problem)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """ Search the node that has the lowest combined cost and heuristic first. """
    fringe = util.PriorityQueue()
    explored = set()

    fringe.push(Node(problem.getStartState()), 0)

    while not fringe.isEmpty():
        node = fringe.pop()
        if problem.isGoalState(node.state): return node.path
        if node.state in explored: continue
        explored.add(node.state)
        for (state, action, cost) in problem.getSuccessors(node.state):
            child = Node(state, node.path + [action], node.cost + cost)
            fringe.push(child, child.cost + heuristic(state, problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
