module Evaluate

type EvaluationPageResult = {Duration:int64;MissingTemplates:int;TooManyTemplates:int;AvgRankingDifference:Option<float>;Precision:float;Recall:float;F1:float}
type EvaluationResult = {AvgDuration:float;MinDurations:int64; MaxDurations:int64;AvgMissingTemplates:float;MinMissingTemplates:int;MaxMissingTemplates:int;AvgTooManyTemplates:float;MinTooManyTemplates:int;MaxTooManyTemplates:int;AvgRankingDifferencePerPage:Option<float>;AvgPrecision:float;AvgRecall:float;AvgF1:float}

let measureTime action =
    let stopWatch = System.Diagnostics.Stopwatch.StartNew()
    let actionResult = action()
    stopWatch.Stop()
    (stopWatch.ElapsedMilliseconds, actionResult)

let evaluate (basicPages : List<Models.BasicPage.Root>) = 
    let basicPagesWithAssignedTemplates = basicPages |> Seq.where (fun x -> x.Templateguids.Length > 0) |> Seq.toList
    let basicPagesLength = basicPagesWithAssignedTemplates.Length
    let evaluate (basicPage : Models.BasicPage.Root) index = 
        printfn "evaluating: %i/%i" index basicPagesLength
        let executeWebRequest = WebRequests.buildWebRequest basicPage
        let duration, response = measureTime executeWebRequest

        let predictedRanking = WebRequests.deserializeResponse response
        let predictedGuids = predictedRanking |> List.map (fun x-> x.TemplateGuid)
        let expectedValues = basicPage.Templateguids |> Seq.mapi (fun i x -> (x, i)) |> readOnlyDict
        
        let missingTemplates = basicPage.Templateguids |> Seq.except predictedGuids |> Seq.length
        let tooManyTemplates = predictedGuids |> Seq.except basicPage.Templateguids |> Seq.length 
        let rankingDifferences = 
            predictedRanking |> 
            Seq.map (fun x -> 
                let containsGuid, expectedRanking = expectedValues.TryGetValue(x.TemplateGuid)
                if containsGuid then Some(abs (expectedRanking - x.Ranking)) else None) |>
            Seq.choose id |>
            Seq.toList
        let avgWrongRanking = if rankingDifferences.Length = 0 then None else Some(rankingDifferences |> Seq.averageBy float)

        let correctlyIdentified = float (expectedValues.Count - missingTemplates)
        let tp_fp = (correctlyIdentified + float tooManyTemplates)
        let precision = if tp_fp <> 0.0 then correctlyIdentified / tp_fp else 1.0

        let tp_fn = (correctlyIdentified + float missingTemplates)
        let recall = if tp_fn = 0.0 then 1.0 else correctlyIdentified / tp_fn
        let precisionAndRecall = (precision + recall)
        let f1 = if precisionAndRecall <> 0.0 then 2.0 * precision * recall / precisionAndRecall else 0.0

        {Duration=duration;MissingTemplates=missingTemplates;TooManyTemplates=tooManyTemplates;AvgRankingDifference=avgWrongRanking;Precision=precision;Recall=recall;F1=f1}

    let evaluations = basicPagesWithAssignedTemplates |> List.mapi (fun i x -> evaluate x i)
    
    let avgDurations = evaluations |> Seq.averageBy (fun x -> float x.Duration)
    let minDurations = evaluations |> Seq.map (fun x -> x.Duration) |> Seq.min
    let maxDurations = evaluations |> Seq.map (fun x -> x.Duration) |> Seq.max
    let avgMissing = evaluations |> Seq.averageBy (fun x -> float x.MissingTemplates)
    let minMissing= evaluations |> Seq.map (fun x -> x.MissingTemplates) |> Seq.min
    let maxMissing = evaluations |> Seq.map (fun x -> x.MissingTemplates) |> Seq.max
    let avgTooMany = evaluations |> Seq.averageBy (fun x -> float x.TooManyTemplates)
    let minTooMany = evaluations |> Seq.map (fun x -> x.TooManyTemplates) |> Seq.min
    let maxTooMany = evaluations |> Seq.map (fun x -> x.TooManyTemplates) |> Seq.max
    let avgPrecision = evaluations |> Seq.averageBy (fun x -> x.Precision)
    let avgRecall = evaluations |> Seq.averageBy (fun x -> x.Recall)
    let avgF1 = evaluations |> Seq.averageBy (fun x -> x.F1)

    let allRankingDifferences = evaluations |> Seq.map (fun x -> x.AvgRankingDifference) |> Seq.choose id |> Seq.toList
    let avgRankingDifferencePerPage = if allRankingDifferences.Length = 0 then None else Some(allRankingDifferences |> Seq.average)
    {
        AvgDuration=avgDurations;
        MinDurations = minDurations;
        MaxDurations=maxDurations;
        AvgMissingTemplates=avgMissing;
        MinMissingTemplates = minMissing;
        MaxMissingTemplates=maxMissing;
        AvgTooManyTemplates=avgTooMany;
        MinTooManyTemplates = minTooMany;
        MaxTooManyTemplates=maxTooMany;
        AvgRankingDifferencePerPage=avgRankingDifferencePerPage;
        AvgPrecision=avgPrecision;
        AvgRecall=avgRecall;
        AvgF1=avgF1}