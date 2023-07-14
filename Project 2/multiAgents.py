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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

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
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        "*** YOUR CODE HERE ***"
        #print(currentGameState.getGhostPositions())
        #Checks ghost location and pacmans next possible move's location
        evalGhost = 1
        evalFood = -1

        # print(newScaredTimes) always 0 ???

        for ghosts in currentGameState.getGhostPositions():
            if abs(newPos[0] - ghosts[0]) + abs(newPos[1] - ghosts[1]) is 1 and newScaredTimes is 0:
                evalGhost = -9999 # if it is too close, eval function should return small number (negative preferred)

        for foods in list(newFood.asList()):
            if abs(newPos[0] - foods[0]) + abs(newPos[1] - foods[1]) < evalFood or evalFood is -1:
                evalFood = abs(newPos[0] - foods[0]) + abs(newPos[1] - foods[1])

        #print(successorGameState.getScore())
        return successorGameState.getScore()  + evalGhost/evalFood

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 0)[0]  # we know first agent is pacman

    def max_value(self, gameState, depth):
        if (self.depth is depth) or gameState.isWin() is not gameState.isLose():  # if win != lose then game is going
            return [None, self.evaluationFunction(gameState)]

        # Same as lecture notes
        v = -9999
        action = None

        for successor in gameState.getLegalActions(0):
            tmp = self.min_value(gameState.generateSuccessor(0, successor), 1, depth)
            if tmp > v:
                v = tmp
                action = successor  # action is needed for the next move therefore we did this

        return [action, v]

    def min_value(self, gameState, agent, depth):

        if gameState.isWin() is not gameState.isLose():  ## if win != lose then game is going
            return self.evaluationFunction(gameState)

        # Same as lecture notes
        v = 9999

        for successor in gameState.getLegalActions(agent):
            if agent == gameState.getNumAgents() - 1:
                v = min(v, self.max_value(gameState.generateSuccessor(agent, successor), depth + 1)[1])
            else:
                v = min(v, self.min_value(gameState.generateSuccessor(agent, successor),(agent + 1) % gameState.getNumAgents(), depth))

        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 0, -9999, 9999)[0]  # we know first agent is pacman

    def max_value(self, gameState, depth, a, b):
        if (self.depth is depth) or gameState.isWin() is not gameState.isLose():  # if win != lose then game is going
            return [None, self.evaluationFunction(gameState)]

        # Same as lecture notes
        v = -9999
        action = None

        for successor in gameState.getLegalActions(0):
            tmp = self.min_value(gameState.generateSuccessor(0, successor), 1, depth, a, b)
            if tmp > v:
                v = tmp
                action = successor  # action is needed for the next move therefore we did this

            if v > b:
                return [action, v]

            a = max(a, v)

        return [action, v]

    def min_value(self, gameState, agent, depth, a, b):

        if gameState.isWin() is not gameState.isLose():  # if win != lose then game is going
            return self.evaluationFunction(gameState)

        # Same as lecture notes
        v = 9999

        for successor in gameState.getLegalActions(agent):
            if agent == gameState.getNumAgents() - 1:
                v = min(v, self.max_value(gameState.generateSuccessor(agent, successor), depth + 1, a, b)[1])
            else:
                v = min(v, self.min_value(gameState.generateSuccessor(agent, successor), (agent + 1) % gameState.getNumAgents(), depth, a, b))

            if v < a:
                return v

            b = min(b, v)

        return v


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
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 0)[0]  # we know first agent is pacman

    def max_value(self, gameState, depth):
        if ( self.depth is depth) or gameState.isWin() is not gameState.isLose():  # if win != lose then game is going
            return [None, self.evaluationFunction(gameState)]

        # Same as lecture notes
        v = -9999
        action = None

        for successor in gameState.getLegalActions(0):
            tmp = self.min_value(gameState.generateSuccessor(0, successor), 1, depth)
            if tmp > v:
                v = tmp
                action = successor  # action is needed for the next move therefore we did this

        return [action, v]

    def min_value(self, gameState, agent, depth):

        if gameState.isWin() is not gameState.isLose():  ## if win != lose then game is going
            return self.evaluationFunction(gameState)

        # Same as lecture notes
        v = 0

        for successor in gameState.getLegalActions(agent):
            p = 1.0 / len(gameState.getLegalActions(agent))
            if agent == gameState.getNumAgents() - 1:
                v = v + p * self.max_value(gameState.generateSuccessor(agent, successor), depth + 1)[1]
            else:
                v = v + p * self.min_value(gameState.generateSuccessor(agent, successor), (agent + 1) % gameState.getNumAgents(), depth)

        return v



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

        In this method, it checks whether ghost is too close or not and if it too close it returns very less score
        Also checks the location of foods, and tries to get closer to the closest food.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    evalGhost = 1
    evalFood = -1

    for ghosts in currentGameState.getGhostPositions():
        if abs(newPos[0] - ghosts[0]) + abs(newPos[1] - ghosts[1]) is 1:
            evalGhost = -9999  # if it is too close, eval function should return small number (negative preferred)

    for foods in list(currentGameState.getFood().asList()):
        if abs(newPos[0] - foods[0]) + abs(newPos[1] - foods[1]) < evalFood or evalFood is -1:
            evalFood = abs(newPos[0] - foods[0]) + abs(newPos[1] - foods[1])

    return currentGameState.getScore() + evalGhost / evalFood

# Abbreviation
better = betterEvaluationFunction
