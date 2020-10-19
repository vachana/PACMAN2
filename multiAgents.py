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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        newGhostStates = successorGameState.getGhostStates()

        # Set the smallest distance as a very small number
        smallestDistance = float('-inf')
        # Get the current food left on the grid and place it inside a list of boolean values
        currFood = currentGameState.getFood()
        foodList = currFood.asList()
        # Run through each remaining food dot
        for food in foodList:
            # Find the distance between the current food dot and Pacman's position
            # Set the distance to negative since we will be calling max(scores) in getAction()
            distance = -(manhattanDistance(food, newPos))
            # We are not interested in stopping, so set score to very small number and eliminate its possibility
            if action == 'Stop':
                distance = float('-inf')
            # If the distance is greater, that means that it is smaller due to the negative,
            # so set it to smallestDistance
            # Remembering that the initial smallestDistance is -inf, so the first distance will always have a
            # greater value
            if distance > smallestDistance:
                smallestDistance = distance

        # Run through each ghost on the grid
        for ghost in newGhostStates:
            # Find the position of the current ghost
            ghostPos = ghost.getPosition()
            # Find the distance between the current ghost and Pacman
            safetySpace = manhattanDistance(ghostPos, newPos)
            # If the distance is less than one, our safetySpace test fails and we set the value to a very small number
            if safetySpace < 1:
                smallestDistance = float('-inf')

        return smallestDistance


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        # Call helper function for the MiniMax Agent
        action, score = self.miniMax(gameState, 0, 0)
        # Returns the best minimax action
        return action

    def miniMax(self, state, depth, agent):

        # Condition to check if we have gone through each agent for the current depth
        # If so, update the current agent to Pacman and increase the depth by 1
        if agent >= state.getNumAgents():
            agent = 0
            depth = depth + 1

        # If we have reached the bottom of the tree or a game-ending state, return the relevant action
        if depth == self.depth or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)

        # Variable that will hold the best action for the agent
        goTo = None
        # Conditional statement if the agent is Pacman
        if agent == 0:
            # Set the max value as a very small number
            maxValue = float('-inf')
            # Run through each legal action that Pacman can take
            for action in state.getLegalActions(agent):
                # Find the state of the agent's successor with the relevant current action
                successorState = state.generateSuccessor(agent, action)
                # Recursively call miniMax for the rest of the agents in the current depth of the tree
                score = self.miniMax(successorState, depth, agent + 1)
                # If the score returned (score[1]) is greater than the max value, set it as the new max value
                # And also update goTo with the action that leads to this score
                if score[1] > maxValue:
                    maxValue = score[1]
                    goTo = action
            return goTo, maxValue
        # Conditional statement if the agent is a Ghost
        else:
            # Set the min value as a very large number
            minValue = float('inf')
            # Run through each legal action that the ghost can take
            for action in state.getLegalActions(agent):
                # Find the state of the agent's successor with the relevant current action
                successorState = state.generateSuccessor(agent, action)
                # Recursively call miniMax for the rest of the agents in the current depth of the tree
                score = self.miniMax(successorState, depth, agent + 1)
                # If the score returned (score[1]) is less than the min value, set it as the new min value
                # And also update goTo with the action that leads to this score
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

        # Call helper function for the MiniMaxAB Agent
        action, score = self.miniMaxAB(gameState, 0, 0, float('-inf'), float('inf'))
        # Return the best minimax action
        return action

    def miniMaxAB(self, state, depth, agent, alpha, beta):

        # Conditional to check if we have gone through each agent for the current depth
        # If so, update the current agent to Pacman and increase the depth by 1
        if agent >= state.getNumAgents():
            agent = 0
            depth = depth + 1

        # If we have reached the bottom of the tree or a game-ending state, return the relevant action
        if depth == self.depth or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)

        # Variable that will hold the best action for the agent
        goTo = None
        # Conditional statement if the agent is Pacman
        if agent == 0:
            # Set the max value as a very small number
            maxValue = float('-inf')
            # Run through each legal action that Pacman can take
            for action in state.getLegalActions(agent):
                # Find the state of the agent's successor with the relevant current action
                successorState = state.generateSuccessor(agent, action)
                # Recursively call miniMaxAB for the rest of the agents in the current depth of the tree
                score = self.miniMaxAB(successorState, depth, agent + 1, alpha, beta)
                # If the score returned (score[1]) is greater than the max value, set it as the new max value
                # And also update goTo with the action that leads to this score
                if score[1] > maxValue:
                    maxValue = score[1]
                    goTo = action
                # This is the alpha-beta pruning section
                # If the max value is greater than beta, return goTo and maxValue before continuing to iterate
                # Through the loop. This is more efficient because of the reduced iterations in the loop
                if maxValue > beta:
                    return goTo, maxValue
                # Update the value of alpha
                alpha = max(alpha, maxValue)
            return goTo, maxValue
        # Conditional statement if the agent is a Ghost
        else:
            # Set the max value as a very large number
            minValue = float('inf')
            # Run through each legal action that the current ghost can take
            for action in state.getLegalActions(agent):
                # Find the state of the agent's successor with the relevant current action
                successorState = state.generateSuccessor(agent, action)
                # Recursively call miniMaxAB for the rest of the agents in the current depth of the tree
                score = self.miniMaxAB(successorState, depth, agent + 1, alpha, beta)
                # If the score returned (score[1]) is less than the min value, set it as the new min value
                # And also update goTo with the action that leads to this score
                if score[1] < minValue:
                    minValue = score[1]
                    goTo = action
                # This is the alpha-beta pruning section
                # If the min value is less than alpha, return goTo and maxValue before continuing to iterate
                # Through the loop. This is more efficient because of the reduced iterations in the loop
                if minValue < alpha:
                    return goTo, minValue
                # Update the value of beta
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

        # Call helper function for the ExpectiMax Agent
        action, score = self.expectiMax(gameState, 0, 0)
        # Return the best minimax action
        return action

    def expectiMax(self, state, depth, agent):

        # Condition to check if we have gone through each agent for the current depth
        # If so, update the current agent to Pacman and increase the depth by 1
        if agent >= state.getNumAgents():
            agent = 0
            depth = depth + 1

        # If we have reached the bottom of the tree or a game-ending state, return the relevant action
        if depth == self.depth or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)

        # Variable that will hold the best action for the agent
        goTo = None
        # Conditional statement if the agent is Pacman
        if agent == 0:
            # Set the max value as a very small number
            maxValue = float('-inf')
            # Run through each legal action that Pacman can take
            for action in state.getLegalActions(agent):
                # Find the state of the agent's successor with the relevant current action
                successorState = state.generateSuccessor(agent, action)
                # Recursively call expectiMax for the rest of the agents in the current depth of the tree
                score = self.expectiMax(successorState, depth, agent + 1)
                # If the score returned (score[1]) is greater than the max value, set it as the new max value
                # And also update goTo with the action that leads to this score
                if score[1] > maxValue:
                    maxValue = score[1]
                    goTo = action
            return goTo, maxValue
        # Conditional statement if the agent is a Ghost
        else:
            # Set the min value as 0.0. This will allow us to create a final sum for the minimum value
            minValue = 0.0
            # Find the total number of actions that the ghost is able to take from the current state
            numActions = len(state.getLegalActions(agent))
            # Run through each legal action that the current ghost can take
            for action in state.getLegalActions(agent):
                # Find the state of the agent's successor with the relevant current action
                successorState = state.generateSuccessor(agent, action)
                # Recursively call expectiMax for the rest of the agents in the current depth of the tree
                _, score = self.expectiMax(successorState, depth, agent + 1)
                # Update the min value with the probability of the current action taking place
                minValue += score / numActions
                # Update best action that the ghost can take
                goTo = action
            return goTo, minValue


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: We created huge incentive for Pacman to take the next action that wins the game and to refrain as much
    as possible from an action that would lose the game. We also added points to the score if Pacman eats the nearest
    food dot and we take points away from the score for a smaller distance between himself and a ghost (thus, Pacman
    would like to stay as far away as possible)
    """

    # If the result is a win, return a very large number creating lots of incentive to move there
    if currentGameState.isWin():
        return float('inf')

    # If the result is a loss, return a very small number creating lots of incentive to move there
    if currentGameState.isLose():
        return float('-inf')

    # Find the current score of the game, Pacman's current position, as well as the food remaining on the grid and
    # place the values in a list as boolean values
    score = currentGameState.getScore()
    foodLeft = currentGameState.getFood()
    foodList = foodLeft.asList()
    pacmanPos = currentGameState.getPacmanPosition()
    capsules = currentGameState.getCapsules()

    # Set the min distance as a very large number
    minDist = float('inf')
    closeDist = float('inf')
    # Run through each of the food dots remaining on the grid
    for food in foodList:
        # Find the distance between Pacman's current position and the current food dot
        dist = util.manhattanDistance(food, pacmanPos)
        # If the distance found above is smaller than minDist, set minDist to dist
        if dist < minDist:
            minDist = dist

    # Run through each of the Power pellets remaining on the grid
    for pellet in capsules:
        # Find the distance between Pacman's current position and the current power pellet
        dist = util.manhattanDistance(pellet, pacmanPos)
        # If the distance found above is smaller than closeDist, set closeDist to dist
        if dist < closeDist:
            closeDist = dist

    # Set the ghost distance as a very large number
    ghostDist = float('inf')
    # Run through each of the ghosts remaining on the grid
    for ghost in currentGameState.getGhostPositions():
        # Find the distance between Pacman's current position and the current ghost
        dist = util.manhattanDistance(ghost, pacmanPos)
        # If the distance found above is smaller than ghostDist, set ghostDist to dist
        if ghostDist > dist:
            ghostDist = dist

    # Create incentive for Pacman to take the action that eats the closest food dot while increasing the distance
    # between himself and other ghosts
    return score + (1 / float(minDist)) + (1 / float(closeDist)) - (1 / float(ghostDist))


# Abbreviation
better = betterEvaluationFunction
