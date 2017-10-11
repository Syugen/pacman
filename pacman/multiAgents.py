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

import datetime

from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

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
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth)[1]

    def minimax(self, state, agentIndex, depth):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth if agentIndex < state.getNumAgents() - 1 else (depth - 1)
        legal = state.getLegalActions(agentIndex)
        successors = [(state.generateSuccessor(agentIndex, action), action) for action in legal]    
        scored = [(self.minimax(state, nextAgent, nextDepth)[0], action) for state, action in successors]
        bestScore = max(scored)[0] if agentIndex == 0 else min(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return bestScore, bestActions[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alpha_beta(gameState, 0, self.depth, (-10e8, None), (10e8, None))[1]

    def alpha_beta(self, state, agentIndex, depth, alpha, beta):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth if agentIndex < state.getNumAgents() - 1 else (depth - 1)
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            result = self.alpha_beta(successor, nextAgent, nextDepth, alpha, beta)
            if agentIndex == 0 and result[0] > alpha[0]: alpha = (result[0], action)
            if agentIndex != 0 and result[0] < beta[0]: beta = (result[0], action)
            if beta[0] <= alpha[0]: break
        return alpha if agentIndex == 0 else beta    

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
        return self.expectimax(gameState, 0, self.depth)[1]

    def expectimax(self, state, agentIndex, depth):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None   
        
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth if agentIndex < state.getNumAgents() - 1 else (depth - 1)
        best = (-10e8, None) if agentIndex == 0 else 0
        legal = state.getLegalActions(agentIndex)
        for action in legal:
            successor = state.generateSuccessor(agentIndex, action)
            score = self.expectimax(successor, nextAgent, nextDepth)[0]
            if agentIndex == 0 and score > best[0]: best = score, action
            if agentIndex != 0: best += score
        if agentIndex != 0: best /= len(legal)
        return best if agentIndex == 0 else (best, None)          


# the following class corrects and replaces the previous MonteCarloAgent class released on March 19
# the only differences between this version, and the one released on March 19 are:
#       * line 37 of this file, "if self.Q" has been replaced by "if Q"
#       * line 45 of this file, where "assert( Q == 'contestClassic' )" has been added
class MonteCarloAgent(MultiAgentSearchAgent):
    """
        Your monte-carlo agent (question 5)
        ***UCT = MCTS + UBC1***
        TODO:
        1) Complete getAction to return the best action based on UCT.
        2) Complete runSimulation to simulate moves using UCT.
        3) Complete final, which updates the value of each of the states visited during a play of the game.

        * If you want to add more functions to further modularize your implementation, feel free to.
        * Make sure that your dictionaries are implemented in the following way:
            -> Keys are game states.
            -> Value are integers. When performing division (i.e. wins/plays) don't forget to convert to float.
      """

    def __init__(self, evalFn='mctsEvalFunction', depth='-1', timeout='50', numTraining=100, C='2', Q=None):
        # This is where you set C, the depth, and the evaluation function for the section "Enhancements for MCTS agent".
        if Q:
            if Q == 'minimaxClassic':
                self.C = 2
            elif Q == 'testClassic':
                self.C = 1.414
                depth = 1
                evalFn='better'
            elif Q == 'smallClassic':
                self.C = 1.414
                depth = 1
                evalFn='better'
            else: # Q == 'contestClassic'
                assert( Q == 'contestClassic' )
                self.C = 1.414
                depth = 1
                evalFn='better'
        # Otherwise, your agent will default to these values.
        else:
            self.C = int(C)
            # If using depth-limited UCT, need to set a heuristic evaluation function.
            if int(depth) > 0:
                evalFn = 'scoreEvaluationFunction'
        self.states = []
        self.plays = dict()
        self.wins = dict()
        self.calculation_time = datetime.timedelta(milliseconds=int(timeout))

        self.numTraining = numTraining

        "*** YOUR CODE HERE ***"

        MultiAgentSearchAgent.__init__(self, evalFn, depth)

    def update(self, state):
        """
        You do not need to modify this function. This function is called every time an agent makes a move.
        """
        self.states.append(state)

    def getAction(self, gameState):
        """
        Returns the best action using UCT. Calls runSimulation to update nodes
        in its wins and plays dictionary, and returns best successor of gameState.
        """
        "*** YOUR CODE HERE ***"
        games = 0
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            games += 1
            self.run_simulation(gameState)
        legal = gameState.getLegalActions(0)
        successors = [(gameState.generateSuccessor(0, action), action) for action in legal]
        successors = [(successor, action) for successor, action in successors
                      if successor in self.plays and self.plays[successor] != 0]
        values =  list((1.0 * self.wins[successor] / self.plays[successor], action)
                       for successor, action in successors)
        max_val = max(value for value in values if not math.isnan(value[0]))
        return random.choice([(successor, action) for successor, action in successors
                             if max_val[0] == (1.0 * self.wins[successor] / self.plays[successor])])[1]

    def run_simulation(self, state):
        """
        Simulates moves based on MCTS.
        1) (Selection) While not at a leaf node, traverse tree using UCB1.
        2) (Expansion) When reach a leaf node, expand.
        4) (Simulation) Select random moves until terminal state is reached.
        3) (Backpropapgation) Update all nodes visited in search tree with appropriate values.
        * Remember to limit the depth of the search only in the expansion phase!
        Updates values of appropriate states in search with with evaluation function.
        """
        "*** YOUR CODE HERE ***"
        player = 0
        visited_states = [(player, state)]
        depth_limited = self.depth != -1
        depth = self.depth
        expand = True
        while not visited_states[-1][1].isWin() and not visited_states[-1][1].isLose():
            if depth_limited and depth == 0: break
            state = self.UCB1(state, player) # Selection & Simulation
            if expand and state not in self.plays: # Expansion
                expand = False
                self.plays[state] = 0
                self.wins[state] = 0
            visited_states.append((player, state))
            player = (player + 1) % state.getNumAgents()
            if not expand and depth_limited and player == 0: depth -= 1
        
        for player, state in visited_states:
            if state in self.plays: # Not simulated nodes
                self.plays[state] += 1
                eval = self.evaluationFunction(visited_states[-1][1])
                if depth_limited:
                    if player == 0: self.wins[state] += eval
                    if player != 0: self.wins[state] -= eval
                else:
                    if player == 0: self.wins[state] += eval
                    if player != 0: self.wins[state] += (1 - eval)

    def UCB1(self, state, player):
        legal = state.getLegalActions(player)
        successors = [state.generateSuccessor(player, action) for action in legal]
        if all(successor in self.plays for successor in successors):
            N = sum(self.plays[successor] for successor in successors)
            return max(((1.0 * self.wins[successor] / self.plays[successor] +
                        self.C * math.sqrt(math.log(N)) / self.plays[successor]),
                        successor) for successor in successors)[1]
        else:
            return random.choice([s for s in successors if s not in self.plays])

    def final(self, state):
        """
        Called by Pacman game at the terminal state.
        Updates search tree values of states that were visited during an actual game of pacman.
        """
        "*** YOUR CODE HERE ***"
        return
        util.raiseNotDefined()

def mctsEvalFunction(state):
    """
    Evaluates state reached at the end of the expansion phase.
    """
    return 1 if state.isWin() else 0

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (to help improve your UCT MCTS).
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():  return -float("inf")
    elif currentGameState.isWin(): return float("inf")

    position = currentGameState.getPacmanPosition()
    score = scoreEvaluationFunction(currentGameState)

    foods = currentGameState.getFood().asList()
    food_distance = min(util.manhattanDistance(position, food) for food in foods)

    ghosts = currentGameState.getGhostStates()
    ghost_distance = max(5, min(util.manhattanDistance(position, ghost.getPosition()) for ghost in ghosts))

    return score - 1.5 * food_distance - 4 * len(foods) - 2.0 / ghost_distance\
                 - 20 * len(currentGameState.getCapsules())

    
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
 
      This evaluation function is meant for use with adversarial search agents
    """
    return currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

