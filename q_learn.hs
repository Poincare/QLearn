import qualified Data.Vector as V
import Numeric
import Data.List
import System.Random

-- | Data type specifying the parameters and Q table for a particular Q learner. qAlpha is the learning
-- rate associated with each iterative update. qGamma is the discount rate on rewards. qGrid is a matrix
-- (dimension number of states by number of actions) that specifies the Q(s,a) function learned by this
-- Q learner. qEpsilon is a function that maps from the number of iterations left to epsilon for the epsilon
-- greedy strategy (can return 1 uniformly if an epsilon greedy strategy is not wanted).
data QLearner = QLearner {qAlpha::Double, qGamma::Double, qEpsilon::(Int -> Double), 
                          qGrid::V.Vector (V.Vector Double)} 

-- |Wrapper around Int, specifying a state index.
data State = State {getStateValue::Int}

-- |Wrapper around Int, specifying an action index.

data Action = Action {getActionValue::Int} -- |Wrapper around Double, specifying a reward.
data Reward = Reward {getRewardValue::Double}

-- |Data type specifying the environment in which the Q learner operates. envExecute is the function
-- used to execute actions at a particular state, returning the new state and the award associated with
-- the state, action pair. envPossible returns the actions possible at any given
-- state.
data Environment = Environment {envExecute::(State -> Action -> (State, Reward)), 
                                envPossible::(State -> [Action])}  


-- |Given alpha, gamma, the number of states and the maximum number of actions possible at any state, 
-- returns a QLearner initialized with a zero Q-table. 
initQLearner :: Double -> Double -> (Int -> Double) -> Int -> Int -> QLearner
initQLearner alpha gamma epsilon numStates numActions = 
  QLearner alpha gamma epsilon $ createZeroQ numStates numActions

-- |Given the envExecute and envPossible functions, constructs an Environment. This is purely for
-- for uniformity of the API. You are welcome to use the data type constructor "Environment" since
-- they are equivalent.
initEnvironment :: (State -> Action -> (State, Reward)) -> (State -> [Action]) -> Environment
initEnvironment execute possible = Environment execute possible    

unwrapExecute :: (State -> Action -> (State, Reward)) -> Int -> Int -> (Int, Double)
unwrapExecute execute state action = let execRet = execute (State state) (Action action)
                                     in (getStateValue $ fst execRet, getRewardValue $ snd execRet)

unwrapPossible :: (State -> [Action]) -> Int -> [Int] 
unwrapPossible possible state = let possibRet = possible (State state)
                                in map (\x -> getActionValue x) possibRet 
   
-- |Given an Environment, a Q learner and the state the Q Learner is on, returns the Q Learner with an updated Q table
-- and the new state of the Q learner within the Environment. Also takes the number of time steps left for the epsilon 
-- computation.
moveLearner :: Int -> StdGen -> Environment -> QLearner -> State -> ((QLearner, State), StdGen)
moveLearner times g (Environment execute' possible') (QLearner alpha gamma epsilon qtable) (State s) =
  let epRet = checkEpsilon g epsilon times
      execute = unwrapExecute execute'
      possible = unwrapPossible possible'
      doRandom = fst $ epRet
      g' = snd $ epRet in
      if doRandom then let randRet = qRandomIter g execute possible s qtable
                           iter = fst randRet
                           g'' = snd randRet
                           qtable' = fst iter
                           state' = snd iter in
                           ((QLearner alpha gamma epsilon qtable', State state'), g'')  
                  else let iter = qLearnIter execute possible s qtable
                           qtable' = fst iter
                           state' = snd iter in
                           ((QLearner alpha gamma epsilon qtable', State state'), g') 
  
-- |Returns the maximum number of characters needed to "show" an element from the given vector.
maxSpaceRow :: V.Vector Double -> Int
maxSpaceRow vec = if V.null vec 
  then 0
  else max (length $ showGFloat (Just 2) (V.head vec) "") (maxSpaceRow $ V.tail vec) 

-- |Returns the maximum number of characters needed to "show" an element in the 2D matrix given.
maxSpaceMat :: V.Vector (V.Vector Double) -> Int
maxSpaceMat mat = if V.null mat
  then 0
  else max (maxSpaceRow $ V.head mat) (maxSpaceMat $ V.tail mat)

-- |Internal function that pads strings with spaces in order to make sure that the string is of a certain length.
padSpaces :: Int -> String -> String
padSpaces space str = str ++ replicate (space - (length str)) ' ' 

-- |Internal function that does a pretty print for a row vector given the maximum space that the
-- row can take up in terms of the characters.
prettyPrintRow :: Int -> V.Vector Double -> String
prettyPrintRow space row = if V.null row
  then ""
  else (padSpaces space $ showGFloat (Just 2) (V.head row) "") ++ "  " ++ (prettyPrintRow space $ V.tail row)

-- |Internal function that does a pretty print for the Q-table given the maximum space that the
-- a single element can take up in terms of characters.
prettyPrintQ' ::  Int -> V.Vector (V.Vector Double) -> String
prettyPrintQ' space mat = if V.null mat
  then ""
  else (prettyPrintRow space $ V.head mat) ++ "\n" ++ (prettyPrintQ' space $ V.tail mat)

-- |Does a pretty print for the Q-table.
prettyPrintQ :: V.Vector (V.Vector Double) -> String
prettyPrintQ mat = let space = maxSpaceMat mat in prettyPrintQ' space mat
 
-- |Create a table for Q(s,a) values, each element representing the expected value of a give state and action
-- pair. Takes the number of possible states and the number of actions as arguments.
createZeroQ :: Int -> Int -> V.Vector (V.Vector Double) 
createZeroQ s a = V.generate s (\n -> V.replicate a 0.0) 

updateQRow :: Int -> Double -> V.Vector Double -> V.Vector Double
updateQRow index value q_row = q_row V.// [(index, value)]

indexQ :: Int -> Int -> V.Vector (V.Vector Double) -> Double
indexQ s a q = q V.! s V.! a 

multIndex row (index:indices) = (row V.! index) : []

unwrapMaybe (Just a) = a
unwrapMaybe Nothing = 0

-- |Figures out the highest Q(s,a) action given a particular state and returns that action index.
maxAction :: (Int -> [Int]) -> Int -> V.Vector (V.Vector Double) -> Int
maxAction possible s q = let possibleActions = possible s
                             possibleValues = map (\action -> q V.! s V.! action) possibleActions
                         in possibleActions !! (unwrapMaybe $ elemIndex (maximum possibleValues) possibleValues)  

randomAction :: StdGen -> (Int -> [Int]) -> Int -> V.Vector (V.Vector Double) -> (Int, StdGen)
randomAction g possible s q = let possibleActions = possible s
                                  randomRet = randomR (0, length possibleActions - 1) g in
                                  (possibleActions !! (fst randomRet), snd randomRet) 
 
-- |Returns the largest Q(s,a) value given a particular state.
maxActionValue :: Int -> V.Vector (V.Vector Double) -> Double
maxActionValue s q = V.maximum (q V.! s)

-- |Updates the Q(s, a) value based on the previous value of Q(s, a) for a given value of s (the state at which an action was executed),
-- a (the action executed at that state), r (the reward attained given the state action pair), s' (the new state) and gamma (the discount
-- factor for the rewards). 
updatedQ :: Int -> Int -> Double -> Int -> Double -> Double -> V.Vector (V.Vector Double) -> V.Vector (V.Vector Double) 
updatedQ s a r s' gamma alpha q = q V.// [(s, updateQRow a updatedValue $ q V.! s)] where
  updatedValue = (indexQ s a q) + alpha * (r + gamma * (maxActionValue s' q) - (indexQ s a q))  

createRewardTable :: Int -> Int -> V.Vector (V.Vector Double) 
createRewardTable s a = V.generate s (\n -> V.replicate a 0.0) 

-- |Create an s x s grid consisting of rewards. Used for grid searches.
createGrid :: Int -> V.Vector (V.Vector Double)
createGrid s = createRewardTable s s

-- |Take a Q table, current state and return the new Q table along with the new state index. Takes a function
-- "execute" that takes a state, action pair and returns the reward and new state associated that state and action pair. 
-- The argument "possible" is a function that gives us a list of actions that are possible at a particular state. For example,
-- we can't go off the grid when we're at the edge of a grid so such an action would not be part of the possible states.
-- TODO make params tunable
qLearnIter :: (Int -> Int -> (Int, Double)) -> (Int -> [Int]) -> Int -> V.Vector (V.Vector Double) -> (V.Vector (V.Vector Double), Int) 
qLearnIter execute possible state q = let action = maxAction possible state q
                                          retExec = execute state action
                                          state' = fst retExec
                                          reward = snd retExec in (updatedQ state action reward state' 0.8 0.4 q, state')

qRandomIter :: StdGen -> (Int -> Int -> (Int, Double)) -> (Int -> [Int]) -> Int -> V.Vector (V.Vector Double) -> ((V.Vector (V.Vector Double), Int), StdGen)
qRandomIter g execute possible state q = let randomRet = randomAction g possible state q
                                             action = fst randomRet
                                             g' = snd randomRet
                                             retExec = execute state action
                                             reward = snd retExec
                                             state' = fst retExec in ((updatedQ state action reward state' 0.8 0.4 q, state'), g')

-- |Takes an integer the width and height of a 2D matrix and a linear index and converts it to a 2D index.
linearTo2D :: Int -> Int -> Int -> (Int, Int)
linearTo2D rows cols lin_index = (lin_index `div` cols, (lin_index `mod` cols))  

-- |Takes a 2D coordinate and turns it into a linear coordinate.
twoDToLinear :: Int -> Int -> (Int, Int) -> Int 
twoDToLinear rows cols (r, c) = (r * cols) + c

-- |Takes the number of rows, number of cols (in a grid), the currents state (specified as a linear index)
-- and an action to determine the next state' (also a linear index). The action can be one of the following:
-- 0: move up
-- 1: move down
-- 2: move left
-- 3: move right.
-- Note that this does not perform any bounds checking. In addition, if the action is invalid, a -1 state is returned.
applyGridAction :: Int -> Int -> Int -> Int -> Int
applyGridAction rows cols state 0 = let state2DIndex = linearTo2D rows cols state
                                        state2DIndex' = (fst state2DIndex - 1, (snd state2DIndex)) 
                                    in twoDToLinear rows cols state2DIndex'

applyGridAction rows cols state 1 = let state2DIndex = linearTo2D rows cols state
                                        state2DIndex' = (fst state2DIndex + 1, snd state2DIndex) 
                                    in twoDToLinear rows cols state2DIndex'

applyGridAction rows cols state 2 = let state2DIndex = linearTo2D rows cols state
                                        state2DIndex' = (fst state2DIndex, snd state2DIndex - 1) 
                                    in twoDToLinear rows cols state2DIndex'

applyGridAction rows cols state 3 = let state2DIndex = linearTo2D rows cols state
                                        state2DIndex' = (fst state2DIndex, snd state2DIndex + 1) 
                                    in twoDToLinear rows cols state2DIndex'

applyGridAction rows cols state _ = -1 

-- |Takes a grid of reward values (i.e. each point in this grid is a state and each state has a reward associated with it)
-- and functions as an "execute" for qLearnIter.  
executeOnGrid :: V.Vector (V.Vector Double) -> Int -> Int -> (Int, Double)
executeOnGrid grid state action = let rows = V.length $ grid
                                      cols = V.length $ (grid V.! 0) 
                                      coord = linearTo2D rows cols state
                                      reward = grid V.! (fst coord) V.! (snd coord) 
                                      state' = applyGridAction rows cols state action
                                  in (state', reward)

gridFromList :: [[Double]] -> V.Vector (V.Vector Double)
gridFromList (list:[]) = V.fromList [V.fromList list]
gridFromList (list:lists) = V.cons (V.fromList list) (gridFromList lists) 

testGrid :: V.Vector (V.Vector Double) 
testGrid = gridFromList [[1.0,2.0,3.0,4.0],
                        [5.0,6.0,7.0,8.0],
                        [12.0,11.0,10.0,9.0],
                        [13.0,14.0,15.0,16.0]]

gridPossibleX i j rows cols
  | j <= 0 = [3]
  | j >= rows-1 = [2]
  | otherwise = [2,3]

gridPossibleY i j rows cols
  | i <= 0 = [1]
  | i >= cols-1 = [0]
  | otherwise = [0, 1] 

gridPossible :: V.Vector (V.Vector Double) -> Int -> [Int]
gridPossible grid state = let rows = V.length grid 
                              cols = V.length $ (grid V.! 0)
                              i = fst $ linearTo2D rows cols state
                              j = snd $ linearTo2D rows cols state 
                          in (gridPossibleX i j rows cols) ++ (gridPossibleY i j rows cols)

qPrint grid times s q = do
  putStrLn $ (++) "Original state: " $ show $ s
  let iter = qLearnIter (executeOnGrid grid) (gridPossible grid) s q 
  let qgrid = fst $ iter
  let state = snd $ iter
  putStrLn $ prettyPrintQ $ qgrid 
  putStrLn $ (++) "State: " $ show $ state
  qPrint grid (times - 1) state qgrid

checkEpsilon :: StdGen -> (Int -> Double) -> Int -> (Bool, StdGen)
checkEpsilon g epsilon times = let randRet = randomR (0, 1) g
                                   randVal = fst randRet 
                                   g' = snd randRet in
                               if randVal < (epsilon times) then (True, g') else (False, g')

pick (x, y) v = if v then x else y

qEpsilonPrint :: StdGen -> (Int -> Double) -> V.Vector (V.Vector Double) -> Int -> Int -> V.Vector (V.Vector Double) -> IO () 
qEpsilonPrint g epsilon grid 0 s q = putStrLn "Done!" 
qEpsilonPrint g epsilon grid times s q = do
  let execute = executeOnGrid grid
      possible = gridPossible grid
      epRet = checkEpsilon g epsilon times
      doRandom = fst $ epRet
      g' = snd $ epRet in
      if doRandom 
        then do
          putStrLn "Doing a random action!"
          let randomRet = qRandomIter g' execute possible s q
          let iter = fst randomRet
          let g'' = snd randomRet
          let qgrid = fst $ iter
          let state = snd $ iter
          putStrLn $ prettyPrintQ $ qgrid
          putStrLn $ (++) "State: " $ show $ state
          qEpsilonPrint g'' epsilon grid (times - 1) state qgrid
        else do
          putStrLn "Doing a normal action"
          putStrLn $ (++) "Original state: " $ show $ s
          let iter = qLearnIter (executeOnGrid grid) (gridPossible grid) s q 
          let qgrid = fst $ iter
          let state = snd $ iter
          putStrLn $ prettyPrintQ $ qgrid 
          putStrLn $ (++) "State: " $ show $ state
          qEpsilonPrint g' epsilon grid (times - 1) state qgrid

epsilon :: Int -> Int -> Double 
-- epsilon totalTimes timesLeft = 1.0/(fromIntegral $ (totalTimes - timesLeft))
epsilon totalTimes timesLeft = 1 

main = do
  g <- getStdGen 
  let totalTimes = 500 
  qEpsilonPrint g (epsilon totalTimes) testGrid totalTimes 0 $ createZeroQ 16 4  
