import QLearn
import System.Random
import qualified Data.Vector as V

main = do
  let alpha = 0.4 
      gamma = 1
      totalTime = 1000
      numStates = 16 -- we are operating in a 4x4 grid 
      numActions = 4 -- up, down, left and right
      -- epsilon = (\timeRemaining -> 1.0/(fromIntegral $ totalTime - timeRemaining))
      epsilon = (\x -> 1.0)
      execute = executeGrid testGrid 
      possible = possibleGrid testGrid
      qLearner = initQLearner alpha gamma epsilon numStates numActions 
      environment = initEnvironment execute possible
  g <- newStdGen 
  -- moveLearnerAndPrint totalTime g environment qLearner (State 0) 
  moveLearnerPrintRepeat totalTime g environment qLearner (State 0)
