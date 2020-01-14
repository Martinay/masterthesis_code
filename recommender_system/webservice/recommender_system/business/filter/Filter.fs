module Filter

let filter (strategyFunction:Models.PredictionData -> Models.PredictionResult) (basicPage:Models.PreparedBasicInfos) (templates:List<Models.PreparedTemplates>) =
    let data = Common.buildPredictionData basicPage templates
    let predictionResult = strategyFunction data
    let filteredTemplates = Seq.zip templates predictionResult.Predictions |> Seq.where (fun x -> (snd x) > 0.5) |> Seq.map fst |> Seq.toList
    filteredTemplates