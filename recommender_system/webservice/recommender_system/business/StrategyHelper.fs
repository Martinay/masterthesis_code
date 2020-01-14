module StrategyHelper

let getFilterAndRankingFunction strategy =    
    match strategy with
    | Models.Strategy.LocalPython -> Python.LocalPythonStrategy.predictFilter, Python.LocalPythonStrategy.predictRanking
    | Models.Strategy.Azure -> AzureStrategy.predictFilter, AzureStrategy.predictRanking
    | Models.Strategy.Ibm -> IbmStrategy.predictFilter, IbmStrategy.predictRanking
    | Models.Strategy.MlNet -> MlNetStrategy.predictFilter, MlNetStrategy.predictRanking
    | _ -> failwith "unknown strategy"