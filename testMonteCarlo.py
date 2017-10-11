import unittest
from pacman import *
from multiAgents import *
import util, layout
import textDisplay
import datetime

class TestGetAction(unittest.TestCase):

    def setUp(self):
        self.rules = ClassicGameRules(30)
        self.ghostType = loadAgent('RandomGhost', nographics=True)
        self.rules.quiet = False

    def testTimeOut(self):
        pacman = MonteCarloAgent()
        self.assertEqual(pacman.calculation_time, datetime.timedelta(milliseconds=50))

    def testSelection(self):
        map = layout.getLayout('testClassic')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(2)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        for i in range(5):# pacman cannot die in 5 moves on this layout
            action = pacman.getAction(state)
            state = state.generateSuccessor(0, action)
        stud_act = pacman.getAction(state)
        actions = state.getLegalActions(0)
        max_val = max(float(pacman.wins[state.generateSuccessor(0, a)]) / pacman.plays[state.generateSuccessor(0, a)]
                      for a in actions if state.generateSuccessor(0, a) in pacman.plays
                      and pacman.plays[state.generateSuccessor(0, a)] != 0)
        max_acts = [a for a in actions if state.generateSuccessor(0, a) in pacman.plays
                    and pacman.plays[state.generateSuccessor(0, a)] != 0 and
                    float(pacman.wins[state.generateSuccessor(0, a)]) /
                    pacman.plays[state.generateSuccessor(0, a)] == max_val]
        self.assertIn(stud_act, max_acts)

class TestRunSimulation(unittest.TestCase):

    def setUp(self):
        self.rules = ClassicGameRules(30)
        self.ghostType = loadAgent('RandomGhost', nographics=True)
        self.rules.quiet = False

    def testDepth(self):
        map = layout.getLayout('smallClassic')
        pacman = MonteCarloAgent(depth=1)
        ghosts = [self.ghostType(i + 1) for i in range(1)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        pacman.plays[state] = 1
        pacman.wins[state] = 0
        states = [state]
        for i in range(2):
            states = [s.generateSuccessor(i, a) for s in states for a in s.getLegalActions(i)]
            for s in states:
                pacman.plays[s] = 1
                pacman.wins[s] = 0
        # If pacman searches to correct depth after expansion, will result in one state getting added to plays.
        prev = len([s for s in pacman.plays if pacman.plays[s] != 0])
        pacman.run_simulation(state)
        self.assertEqual(prev + 1, len([s for s in pacman.plays if pacman.plays[s] != 0]))

    def testExpansion(self):
        map = layout.getLayout('smallClassic')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(1)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        for i in range(2):
            pacman.run_simulation(state)
        self.assertEqual(len([s for s in pacman.plays if pacman.plays[s] != 0]), 2)

    def testEarlyWin(self):
        map = layout.tryToLoad('test_cases/shortClassic.lay')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(1)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        actions = state.getLegalActions(0)
        pacman.plays[state] = len(actions) + 1
        pacman.wins[state] = len(actions)
        # Run it once for every successor.
        for a in actions:
            succ = state.generateSuccessor(0, a)
            if a == Directions.WEST:
                pacman.plays[succ] = 1
                pacman.wins[succ] = 1
            else:
                pacman.plays[succ] = 1
                pacman.wins[succ] = 0
        # Based on UCB1 should pick state where will win, if don't exit early, code will crash.
        pacman.run_simulation(state)

    def testNoInfo(self):
        map = layout.getLayout('smallClassic')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(1)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        succ = [state.generateSuccessor(0, a) for a in state.getLegalActions(0)]
        pacman.plays[state] = len(succ)
        pacman.wins[state] = len(succ)
        for s in succ[1:len(succ)]:
            pacman.plays[s] = 1
            pacman.wins[s] = 0
        try:
            pacman.run_simulation(state)
        except KeyError:
            # Using method where expand all successor states and set wins and plays to 0
            pacman.plays[succ[-1]] = 0
            pacman.wins[succ[-1]] = 0
            pacman.run_simulation(state)
        # Should play the single state without information
        self.assertEqual(pacman.plays[succ[-1]], 1)

    def testUCB1(self):
        map = layout.getLayout('smallClassic')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(1)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        succ = [state.generateSuccessor(0, a) for a in state.getLegalActions(0)]
        pacman.plays[state] = len(succ) + 1
        pacman.wins[state] = len(succ)
        pacman.plays[succ[0]] = 1
        pacman.wins[succ[0]] = 1
        for i in range(1, len(succ)):
            pacman.plays[succ[i]] = 1
            pacman.wins[succ[i]] = 0
        pacman.run_simulation(state)
        self.assertEqual(pacman.plays[succ[0]], 2)

    def testUCB2(self):
        map = layout.getLayout('smallClassic')
        pacman = MonteCarloAgent()
        ghosts = [self.ghostType(i + 1) for i in range(1)]
        game = self.rules.newGame(map, pacman, ghosts, textDisplay.NullGraphics(), True, True)
        state = game.state
        succ = [state.generateSuccessor(0, a) for a in state.getLegalActions(0)]
        pacman.plays[state] = len(succ) + 2
        pacman.wins[state] = len(succ) + 2
        pacman.plays[succ[0]] = 1
        pacman.wins[succ[0]] = 0
        for i in range(1, len(succ)):
            pacman.plays[succ[i]] = 2
            pacman.wins[succ[i]] = 0
        pacman.run_simulation(state)
        self.assertEqual(pacman.plays[succ[0]], 2)


if __name__ == '__main__':
    unittest.main()