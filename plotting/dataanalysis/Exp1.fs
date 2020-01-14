module Exp1

open FSharp.Data
open System.IO
open XPlot.Plotly

let fileDirectory = "C:\Users\marti\Desktop\datenV1"

let trainingFilterPath = sprintf "%s/training.csv" fileDirectory
let validationFilterPath = sprintf "%s/validation.csv" fileDirectory
let testFilterPath = sprintf "%s/test.csv" fileDirectory

let trainingRankingPath = sprintf "%s/trainingRanking.csv" fileDirectory
let validationRankingPath = sprintf "%s/validationRanking.csv" fileDirectory
let testRankingPath = sprintf "%s/testRanking.csv" fileDirectory

type RankingPredictions = JsonProvider<"files/experiment1/ranking_predicted.txt">
type RankingPredictionResult = {label:int;min:float;max:float;avg:float}

type FilterPredictions = JsonProvider<"files/experiment1/filter_predicted.txt">
type FilterPredictionResult = {tp:int;tn:int;fp:int;fn:int}

type RankingHistory = JsonProvider<"files/experiment1/ranking_history.json">
type FilterHistory = JsonProvider<"files/experiment1/filter_history.json">

let readLines (filePath:string) = seq {
    use sr = new StreamReader (filePath)
    while not sr.EndOfStream do
        yield sr.ReadLine ()
}

let plotrankingPredictions() =
    let data = RankingPredictions.Load("files/experiment1/ranking_predicted.txt")
    let predictions =
        data |> 
        Seq.groupBy (fun x -> x.Label) |> 
        Seq.map (fun x -> 
            let predictions = (snd x) |> Seq.map (fun y -> y.Prediction) |> Seq.toList
            let predictionsWithoutOutliers = 
                if(predictions.Length > 3)
                then
                    predictions |> Seq.sort |> Seq.take (int(float(predictions.Length) * 0.98)) |> Seq.skip (int(float(predictions.Length) * 0.02)) |> Seq.toList
                else
                predictions
                
            {
            label=fst x
            min=predictionsWithoutOutliers |> Seq.min;
            max=predictionsWithoutOutliers |> Seq.max;
            avg=predictions |> Seq.average
            }) |>
        Seq.sortBy (fun x -> x.label) |>
        Seq.toList

    let x = predictions |> Seq.map (fun x -> x.label) |> Seq.toList
    let lineMin =
        Scatter(
            x = x,
            y = (predictions |> Seq.map (fun x -> x.min)),
            mode = "lines+markers",
            name = "Min",
            marker =
                Marker(
                    color = "rgb(200, 238, 245)"
                )
        )
    let lineMax =
        Scatter(
            x = x,
            y = (predictions |> Seq.map (fun x -> x.max)),
            mode = "markers",
            name = "Max",
            fill = "tonextx",
            marker =
                Marker(
                    color = "rgb(200, 238, 245)"
                )
        )
    let lineAvg =
        Scatter(
            x = x,
            y = (predictions |> Seq.map (fun x -> x.avg)),
            mode = "lines+markers",
            name = "Avg",
            marker =
                Marker(
                    color = "rgb(64, 195, 250)",
                    line =
                        Line(
                            color = "rgb(120, 195, 213)"
                        )
                )
        )
    let lineIdeal =
        Scatter(
            x = x,
            y = x,
            mode = "lines",
            name = "Ideal",
            line = Line(color = "rgb(51, 151, 61)")
        )

    let layout = 
        Layout(
            xaxis = Xaxis(title="Tatsächliche Werte"),
            yaxis = Yaxis(title="Vorhersage"),
            title = "Vorhersage / tatsächliche Werte"
        )

    let plot =
        [lineMin;lineMax;lineAvg;lineIdeal]
        |> Chart.Plot
        |> Chart.WithHeight 400
        |> Chart.WithWidth 800
        |> Chart.WithLayout layout
    Chart.Show plot
    ()
       
let plotFilterPredictions() =
    let data = FilterPredictions.Load("files/experiment1/filter_predicted.txt") |> Seq.toList
    let mutable tp = 0
    let mutable tn = 0
    let mutable fp = 0
    let mutable fn = 0

    data |> 
        Seq.iter (fun x -> 
            if x.Label = true && x.Prediction > 0.5 then
                tp <- tp + 1
            elif x.Label = true && x.Prediction <= 0.5 then
                fn <- fn + 1
            elif x.Label = false && x.Prediction > 0.5 then
                fp <- fp + 1
            elif x.Label = false && x.Prediction <= 0.5 then
                tn <- tn + 1
            )
    
    printfn "tp:%i" tp
    printfn "tn:%i" tn
    printfn "fp:%i" fp
    printfn "fn:%i" fn

    printfn "tp:%f" (float(tp) / float(data.Length))
    printfn "tn:%f" (float(tn) / float(data.Length))
    printfn "fp:%f" (float(fp) / float(data.Length))
    printfn "fn:%f" (float(fn) / float(data.Length))
    ()

let plotExperimentHistoryFilter()=   
    let history = FilterHistory.Load("files/experiment1/filter_history.json")

    let x = seq{1..20}

    //let lineLoss =
    //    Scatter(
    //        x = x,
    //        y = history.Loss,
    //        mode = "lines+markers",
    //        name = "Loss"
    //    )
    let lineAuprc =
        Scatter(
            x = x,
            y = history.ValAuprc,
            mode = "lines+markers",
            name = "Fläche unter dem PR-Diagramm"
        )
    let linePrecision =
        Scatter(
            x = x,
            y = history.ValPrecision,
            mode = "lines+markers",
            name = "Genauigkeit"
        )
    let lineRecall =
        Scatter(
            x = x,
            y = history.ValRecall,
            mode = "lines+markers",
            name = "Trefferquote"
        )

    let layout = 
        Layout(
            xaxis = Xaxis(title="Epoche"),
            title = "Lernkurve"
        )

    let plot =
        [(*lineLoss;*)lineAuprc;linePrecision;lineRecall]
        |> Chart.Plot
        |> Chart.WithHeight 400
        |> Chart.WithWidth 800
        |> Chart.WithLayout layout
    Chart.Show plot
    ()

let plotExperimentHistoryRanking()=   
    let history = RankingHistory.Load("files/experiment1/ranking_history.json")
    let x = seq{1..20}

    let lineLoss =
        Scatter(
            x = x,
            y = history.Loss,
            mode = "lines+markers",
            name = "Kostenfunktion"
        )
    let lineMAE =
        Scatter(
            x = x,
            y = history.ValMeanAbsoluteError,
            mode = "lines+markers",
            name = "Mittlerer absoluter Fehler"
        )
    let lineMSE =
        Scatter(
            x = x,
            y = history.ValRootMeanSquaredError,
            mode = "lines+markers",
            name = "Wurzel der mittleren Fehlerquadratsumme"
        )

    let layout = 
        Layout(
            xaxis = Xaxis(title="Epoche"),
            title = "Lernkurve"
        )

    let plot =
        [lineLoss;lineMAE;lineMSE]
        |> Chart.Plot
        |> Chart.WithHeight 400
        |> Chart.WithWidth 800
        |> Chart.WithLayout layout
    Chart.Show plot
    ()


let countDataFilter() = 
    let generateDataStatisticFilter (filepath:string) =
        let data = readLines filepath
        let matchingdata = data |> Seq.skip 1 |> Seq.map (fun x -> if x.[0] = '1' then 1 else 0) |> Seq.toList
        let matchingCount = matchingdata |> Seq.sum
        let allLength = matchingdata.Length
        let notMatching =  allLength - int(matchingCount)
        printfn "%s" filepath
        printfn "matching %A notMatching %i all %i" matchingCount notMatching allLength
        ()

    [trainingFilterPath;validationFilterPath;testFilterPath]
        |> Seq.iter generateDataStatisticFilter
    ()

let countDataRanking() = 
    let generateDataStatistic (filepath:string) =
        let data = readLines filepath
        let allLength = data |> Seq.skip 1 |> Seq.length 
        printfn "%s" filepath
        printfn "all %i" allLength
        ()

    [trainingRankingPath;validationRankingPath;testRankingPath]
        |> Seq.iter generateDataStatistic
    ()