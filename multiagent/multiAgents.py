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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        manhattan = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1])

        successor_game_state = currentGameState.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood()
        new_ghost_states = successor_game_state.getGhostStates()
        new_ghost_distances = [manhattan(ghostState.configuration.pos, new_pos) for ghostState in new_ghost_states]
        close_ghost_distances = [distance for distance in new_ghost_distances if distance < 5]
        new_food_distances = [manhattan(foodPos, new_pos) for foodPos in new_food.asList()]
        score = successor_game_state.getScore()
        if close_ghost_distances:
            score += min(close_ghost_distances)
        if new_food_distances:
            score += 1.0 / min(new_food_distances)
        return score


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
        number_of_agents = gameState.getNumAgents()

        def max_search(game_state, depth=0):
            if depth == self.depth or game_state.isLose() or game_state.isWin():
                return {'score': self.evaluationFunction(game_state)}
            legal_moves = game_state.getLegalActions(0)

            actions_and_scores = [{'action': action,
                                   'score': min_search(game_state.generateSuccessor(0, action),
                                                       depth, 1)}
                                  for action in legal_moves]
            return max(actions_and_scores, key=lambda _action: _action['score'])

        def min_search(game_state, depth, agent=1):
            legal_moves = game_state.getLegalActions(agent)
            successor_states = [game_state.generateSuccessor(agent,
                                                             move)
                                for move in legal_moves]

            if depth == self.depth or game_state.isLose() or game_state.isWin():
                return self.evaluationFunction(game_state)

            if agent == number_of_agents - 1:
                return min([max_search(state, depth + 1)['score']
                            for state in successor_states])

            return min([min_search(state, depth, agent + 1)
                        for state in successor_states])

        return max_search(gameState)['action']


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        number_of_agents = gameState.getNumAgents()

        def max_search(game_state, depth, alpha, beta):
            if depth == self.depth or game_state.isLose() or game_state.isWin():
                return {'score': self.evaluationFunction(game_state)}
            legal_moves = game_state.getLegalActions(0)
            score = -float('inf')
            return_move = None
            for action in legal_moves:
                temp_score = max(score, min_search(game_state.generateSuccessor(0, action), depth, 1, alpha, beta))
                if temp_score > score:
                    return_move = action
                    score = temp_score
                if score > beta:
                    return {'action': action,
                            'score': score}
                alpha = max(alpha, score)
            return {'action': return_move,
                    'score': score}

        def min_search(game_state, depth, agent, alpha, beta):
            legal_moves = game_state.getLegalActions(agent)

            if depth == self.depth or game_state.isLose() or game_state.isWin():
                return self.evaluationFunction(game_state)

            score = float('inf')
            for move in legal_moves:
                state = game_state.generateSuccessor(agent, move)
                if agent == number_of_agents - 1:
                    temp_score = max_search(state, depth + 1, alpha, beta)['score']
                else:
                    temp_score = min_search(state, depth, agent + 1, alpha, beta)
                score = min(score, temp_score)
                if score < alpha:
                    return score
                beta = min(beta, score)
            return score

        return max_search(gameState, depth=0, alpha=-float('inf'), beta=float('inf'))['action']


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
        number_of_agents = gameState.getNumAgents()

        def max_search(game_state, depth=0):
            if depth == self.depth or game_state.isLose() or game_state.isWin():
                return {'score': self.evaluationFunction(game_state)}
            legal_moves = game_state.getLegalActions(0)

            actions_and_scores = [{'action': action,
                                   'score': expectation_search(game_state.generateSuccessor(0, action),
                                                       depth, 1)}
                                  for action in legal_moves]
            return max(actions_and_scores, key=lambda _action: _action['score'])

        def expectation_search(game_state, depth, agent=1):
            legal_moves = game_state.getLegalActions(agent)
            successor_states = [game_state.generateSuccessor(agent,
                                                             move)
                                for move in legal_moves]

            if depth == self.depth or game_state.isLose() or game_state.isWin():
                return self.evaluationFunction(game_state)

            score = 0.0
            for state in successor_states:
                if agent == number_of_agents - 1:
                    score += max_search(state, depth + 1)['score']
                else:
                    score += expectation_search(state, depth, agent + 1)
            average = float(score)/float(len(successor_states))
            return average

        return max_search(gameState)['action']


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: I take account of the distance to the closest ghost,
      and the reciprocal distance to the closest dot. The number I use are empirical :D
    """
    "*** YOUR CODE HERE ***"
    manhattan = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1])

    new_pos = currentGameState.getPacmanPosition()
    new_food = currentGameState.getFood()
    new_ghost_states = currentGameState.getGhostStates()
    new_ghost_distances = [manhattan(ghostState.configuration.pos, new_pos) for ghostState in new_ghost_states]
    close_ghost_distances = [distance for distance in new_ghost_distances if distance < 5]
    new_food_distances = [manhattan(foodPos, new_pos) for foodPos in new_food.asList()]
    score = currentGameState.getScore()
    if close_ghost_distances:
        score += 0.5*min(close_ghost_distances)
    if new_food_distances:
        score += 0.5 / min(new_food_distances)
    return score

# Abbreviation
better = betterEvaluationFunction
