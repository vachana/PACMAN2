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

        smallestDistance = -100000000
        currFood = currentGameState.getFood()
        foodList = currFood.asList()
        for food in foodList:
            distance = -(manhattanDistance(food, newPos))
            if action == 'Stop':
                distance = -1000000000
            if distance > smallestDistance:
                smallestDistance = distance

        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            safetySpace = manhattanDistance(ghostPos, newPos)
            if safetySpace < 1:
                 smallestDistance = -10000000000

        return smallestDistance

        """score = successorGameState.getScore()
        foodArray = newFood.asList()

        # Taking reciprocal since closer the food higher should be the score returned

        for i in foodArray:
            foodDist = util.manhattanDistance(i, newPos)
            if (foodDist) != 0:
                score = score + (1.0 / foodDist)

        # Taking reciprocal since the score returned with respect to ghost is negative, closer the ghost higher should be the score 		in negative

        for ghost in newGhostStates:
            ghostpos = ghost.getPosition()
            ghostDist = util.manhattanDistance(ghostpos, newPos)
            if (abs(newPos[0] - ghostpos[0]) + abs(newPos[1] - ghostpos[1])) > 1:
                score = score + (1.0 / ghostDist)

        return score"""


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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        action, score = self.miniMax(gameState, 0, 0)
        return action

        #util.raiseNotDefined()

    def miniMax(self, state, depth, agent):

        if agent >= state.getNumAgents():
            agent = 0
            depth = depth + 1

        if depth == self.depth or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)

        if agent == 0:
            maxValue = float('-inf')
            for action in state.getLegalActions(agent):
                successorState = state.generateSuccessor(agent, action)
                score = self.miniMax(successorState, depth, agent + 1)
                if score[1] > maxValue:
                    maxValue = score[1]
                    goTo = action
            return goTo, maxValue
        else:
            minValue = float('inf')
            for action in state.getLegalActions(agent):
                successorState = state.generateSuccessor(agent, action)
                score = self.miniMax(successorState, depth, agent + 1)
                if score[1] < minValue:
                    minValue = score[1]
                    goTo = action
            return goTo, minValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        action, score = self.miniMaxAB(gameState, 0, 0, float('-inf'), float('inf'))
        return action





        #util.raiseNotDefined()

    def miniMaxAB(self, state, depth, agent, alpha, beta):
        if agent >= state.getNumAgents():
            agent = 0
            depth = depth + 1

        if depth == self.depth or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)

        if agent == 0:
            maxValue = -1000000000

            for action in state.getLegalActions(agent):
                successorState = state.generateSuccessor(agent, action)
                score = self.miniMaxAB(successorState, depth, agent + 1, alpha, beta)

                if score[1] > maxValue:
                    maxValue = score[1]
                    goTo = action
                if maxValue > beta:
                    return goTo, maxValue
                alpha = max(alpha, maxValue)
            return goTo, maxValue
        else:
            minValue = 100000000
            for action in state.getLegalActions(agent):
                successorState = state.generateSuccessor(agent, action)
                score = self.miniMaxAB(successorState, depth, agent + 1, alpha, beta)

                if score[1] < minValue:
                    minValue = score[1]
                    goTo = action
                if minValue < alpha:
                    return goTo, minValue
                beta = min(beta, minValue)
            return goTo, minValue


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
        "*** YOUR CODE HERE ***"

        action, score = self.expectiMax(gameState, 0, 0)
        return action

        # util.raiseNotDefined()

    def expectiMax(self, state, depth, agent):

        if agent >= state.getNumAgents():
            agent = 0
            depth = depth + 1

        if depth == self.depth or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)

        if agent == 0:
            maxValue = float('-inf')
            for action in state.getLegalActions(agent):
                successorState = state.generateSuccessor(agent, action)
                score = self.expectiMax(successorState, depth, agent + 1)
                if score[1] > maxValue:
                    maxValue = score[1]
                    goTo = action
            return goTo, maxValue
        else:
            minValue = 0.0
            #prob = 1.0 / len(state.getLegalActions(agent))
            numActions = len(state.getLegalActions(agent))
            for action in state.getLegalActions(agent):
                successorState = state.generateSuccessor(agent, action)
                _, score = self.expectiMax(successorState, depth, agent + 1)
                #if score[1] < minValue:
                minValue += score / numActions
                goTo = action
            return goTo, minValue


        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
