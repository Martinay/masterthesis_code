module Rank

let rank (strategyFunction:Models.PredictionData -> Models.PredictionResult) (basicPage:Models.PreparedBasicInfos) (templates:List<Models.PreparedTemplates>) =
        
    let data = Common.buildPredictionData basicPage templates
    let predictionResult = strategyFunction data
    let rankedTemplates = Seq.zip templates predictionResult.Predictions |> Seq.sortBy snd |> Seq.map fst |> Seq.toList

    rankedTemplates |> List.mapi (fun i x -> {WebRequests.Recommendation.Ranking=i;WebRequests.Recommendation.TemplateGuid=x.Page.Guid})