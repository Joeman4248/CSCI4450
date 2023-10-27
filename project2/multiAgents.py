# multiAgents.py
# --------------
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


import random

import util
import pacman
from game import Agent, Directions, AgentState
from util import manhattanDistance
from pacman import GameState

GHOST_INDEX = 1
PACMAN_INDEX = 0

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Score to add to existing successor score
        scoreChange = 0

        if successorGameState.isLose():
            scoreChange -= 10000

        # Manhattan distance to available foods from successor state
        newFoodList = successorGameState.getFood().asList()
        newFoodDistance = [manhattanDistance(newPos, pos) for pos in newFoodList]
        newFoodCount = len(newFoodList)

        # Manhattan distance to available foods from current state
        foodList = currentGameState.getFood().asList()
        foodDistance = [manhattanDistance(newPos, pos) for pos in foodList]
        foodCount = len(foodList)

        scoreChange -= min(newFoodDistance, default=0) // 2

        scoreChange -= min([manhattanDistance(newPos, pos) for pos in successorGameState.getCapsules()], default=0)

        if newFoodCount < foodCount:
            scoreChange += 200

        if len(successorGameState.getCapsules()) < len(currentGameState.getCapsules()):
            scoreChange += 400

        scoreChange -= 10 * newFoodCount

        # Manhattan distance to ghosts from successor state
        newGhostDistance = [manhattanDistance(newPos, pos) for pos in successorGameState.getGhostPositions()]

        # Manhattan distance to ghosts from current state
        ghostDistance = [manhattanDistance(newPos, pos) for pos in currentGameState.getGhostPositions()]

        # Are ghosts closer or further away in the successor state?
        ghostCloser = min(newGhostDistance, default=0) < min(ghostDistance)
        ghostsScared = sum(newScaredTimes) > 2

        if ghostsScared:
            if ghostCloser:
                scoreChange += 500
            else:
                scoreChange -= 200
        else:
            if ghostCloser:
                scoreChange -= 500
            else:
                scoreChange += 100

        if action == Directions.STOP:
            scoreChange -= 15

        return successorGameState.getScore() + scoreChange


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        maxScore = float('-inf')
        maxAction = None
        depth = 0
        for action in gameState.getLegalActions(PACMAN_INDEX):
            successor = gameState.generateSuccessor(PACMAN_INDEX, action)
            score = self.pacmanScore(successor, depth)
            if score > maxScore:
                maxScore, maxAction = score, action

        return maxAction

    # Used for Pacman
    def pacmanScore(self, gameState: GameState, depth: int, agentIndex=PACMAN_INDEX):
        # Check for terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        maxScore = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            maxScore = max(maxScore, self.GhostScore(successor, depth))

        return maxScore

    # Used for Ghosts
    def ghostScore(self, gameState: GameState, depth: int, agentIndex=GHOST_INDEX):
        # Check for terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        minScore = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            # if last ghost, switch to Pacman, otherwise switch to next ghost
            if agentIndex == (gameState.getNumAgents() - 1):
                minScore = min(minScore, self.pacmanScore(successor, depth+1))
            else:
                minScore = min(minScore, self.GhostScore(successor, depth, agentIndex+1))

        return minScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float('-inf')
        beta = float('inf')
        maxScore = float('-inf')
        maxAction = None
        depth = 0
        for action in gameState.getLegalActions(PACMAN_INDEX):
            successor = gameState.generateSuccessor(PACMAN_INDEX, action)
            score = self.GhostScore(successor, depth, alpha, beta)
            if score > maxScore:
                maxScore, maxAction = score, action
            if maxScore > beta:
                return maxAction
            alpha = max(maxScore, alpha)

        return maxAction

    def pacmanScore(self, gameState: GameState, depth: int, alpha: int, beta: int, agentIndex=PACMAN_INDEX):
        # Check for terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        maxScore = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            maxScore = max(maxScore, self.GhostScore(successor, depth, alpha, beta))

            if maxScore > beta:
                return maxScore
            alpha = max(maxScore, alpha)

        return maxScore

    def GhostScore(self, gameState: GameState, depth: int, alpha: int, beta: int, agentIndex=GHOST_INDEX):
        # Check for terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        minScore = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            # if last ghost, switch to Pacman, otherwise switch to next ghost
            if agentIndex == (gameState.getNumAgents() - 1):
                minScore = min(minScore, self.pacmanScore(successor, depth+1, alpha, beta))
            else:
                minScore = min(minScore, self.GhostScore(successor, depth, alpha, beta, agentIndex+1))

            if minScore < alpha:
                return minScore
            beta = min(minScore, beta)

        return minScore


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        maxScore = float('-inf')
        maxAction = None
        depth = 0
        for action in gameState.getLegalActions(PACMAN_INDEX):
            successor = gameState.generateSuccessor(PACMAN_INDEX, action)
            score = self.GhostScore(successor, depth)
            if score > maxScore:
                maxScore, maxAction = score, action

        return maxAction

    # Used for Pacman
    def pacmanScore(self, gameState: GameState, depth: int, agentIndex=PACMAN_INDEX):
        # Check for terminal state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        maxScore = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            maxScore = max(maxScore, self.GhostScore(successor, depth))

        return maxScore


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
