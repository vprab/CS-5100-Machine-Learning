# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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

        if action is 'Stop':
            return -float("inf")

        for gs in newGhostStates:
            if gs.getPosition() == newPos and gs.scaredTimer is 0:
                return -float("inf")

        currentFoodList = currentGameState.getFood().asList()
        foodDistances = map(lambda f: manhattanDistance(newPos, f), currentFoodList)
        if not foodDistances:
            foodDistances.append(0)

        return successorGameState.getScore() - min(foodDistances)


def scoreEvaluationFunction(currentGameState):
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
    def getAction(self, gameState):
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
        """
        val = self.value(gameState, 0, 0)
        return val[0]

    def value(self, gameState, agentIndex, curDepth):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            curDepth += 1

        if curDepth == self.depth or not gameState.getLegalActions(agentIndex):
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            v = ("unknown", -float("inf"))

            for action in gameState.getLegalActions(agentIndex):
                if action == "Stop":
                    continue

                childVal = self.value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, curDepth)
                if type(childVal) is tuple:
                    childVal = childVal[1]

                vNew = max(v[1], childVal)

                if vNew is not v[1]:
                    v = (action, vNew)

            return v
        else:
            v = ("unknown", float("inf"))

            for action in gameState.getLegalActions(agentIndex):
                if action == "Stop":
                    continue

                childVal = self.value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, curDepth)
                if type(childVal) is tuple:
                    childVal = childVal[1]

                vNew = min(v[1], childVal)

                if vNew is not v[1]:
                    v = (action, vNew)

            return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        val = self.value(gameState, 0, 0, -float("inf"), float("inf"))
        return val[0]

    def value(self, gameState, agentIndex, curDepth, alpha, beta):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            curDepth += 1

        if curDepth == self.depth or not gameState.getLegalActions(agentIndex):
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            v = ("unknown", -float("inf"))

            for action in gameState.getLegalActions(agentIndex):
                if action == "Stop":
                    continue

                childVal = self.value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, curDepth, alpha, beta)
                if type(childVal) is tuple:
                    childVal = childVal[1]

                vNew = max(v[1], childVal)

                if vNew is not v[1]:
                    v = (action, vNew)

                if v[1] > beta:
                    return v

                alpha = max(alpha, v[1])

            return v
        else:
            v = ("unknown", float("inf"))

            for action in gameState.getLegalActions(agentIndex):
                if action == "Stop":
                    continue

                childVal = self.value(gameState.generateSuccessor(agentIndex, action), agentIndex + 1, curDepth, alpha, beta)
                if type(childVal) is tuple:
                    childVal = childVal[1]

                vNew = min(v[1], childVal)

                if vNew is not v[1]:
                    v = (action, vNew)

                if v[1] < alpha:
                    return v

                beta = min(beta, v[1])

            return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        val = self.value(gameState, 0, 0)
        return val[0]

    def value(self, gameState, currentAgentIndex, currentDepth):
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            currentDepth += 1

        if currentDepth == self.depth or not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        if currentAgentIndex == 0:
            v = ("unknown", -float("inf"))

            for action in gameState.getLegalActions(currentAgentIndex):
                if action == "Stop":
                    continue

                retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, currentDepth)
                if type(retVal) is tuple:
                    retVal = retVal[1]

                vNew = max(v[1], retVal)

                if vNew is not v[1]:
                    v = (action, vNew)

            return v
        else:
            div = len(gameState.getLegalActions(currentAgentIndex))
            v = 0.0

            for action in gameState.getLegalActions(currentAgentIndex):
                if action == "Stop":
                    continue

                retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, currentDepth)
                if type(retVal) is tuple:
                    retVal = retVal[1]

                v += retVal

            return v/div

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: This evaluation function values:
        - A high game score
        - Being close to a food pellet
        - Being far from a ghost
        - Eating capsules
    """
    ghostStates = currentGameState.getGhostStates()
    pacmanPos = currentGameState.getPacmanPosition()
    ghostDistances = map(lambda g: manhattanDistance(pacmanPos, g.getPosition()), ghostStates)
    foodDistances = map(lambda f: manhattanDistance(pacmanPos, f), currentGameState.getFood().asList())

    if not ghostDistances:
        ghostDistances.append(0)

    if not foodDistances:
        foodDistances.append(0)

    return currentGameState.getScore() - min(foodDistances) + min(ghostDistances) - 50*len(currentGameState.getCapsules())

# Abbreviation
better = betterEvaluationFunction

