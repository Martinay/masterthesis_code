module ActivationFunctions

open System
open XPlot.Plotly

let sigmoid() =
    let points =  [ for x in -6.0 ..0.1 .. 6.0 -> (x, 1.0 / (1.0 + Math.E ** (-x))) ]
    let plot = 
        points |> Chart.Line

    Chart.Show plot
    ()

let tanh() =
    let points =  [ for x in -6.0 ..0.1 .. 6.0 -> (x, Math.Sinh(x)/Math.Cosh(x)) ]
    let plot = 
        points |> Chart.Line

    Chart.Show plot
    ()
let relu() =
    let points =  [ for x in -6.0 ..0.1 .. 6.0 -> (x, if x < 0.0 then 0.0 else x) ]
    let plot = 
        points |> Chart.Line

    Chart.Show plot
    ()
let leakyrelu() =
    let points =  [ for x in -6.0 ..0.1 .. 6.0 -> (x, if x < 0.0 then x*0.2 else x) ]
    let plot = 
        points |> Chart.Line

    Chart.Show plot
    ()