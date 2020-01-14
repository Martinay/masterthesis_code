// Learn more about F# at http://fsharp.org

open System
open FSharp.Data
open XPlot.Plotly

type Rawtemplate = CsvProvider<"files/raw/Welle5/templatePages.csv","\t">
type Rawbasic = JsonProvider<"files/raw/Welle5/basicPages.json">

let rawTemplate()=
    let rawWelle5 = Rawtemplate.Load("files/raw/Welle5/templatePages.csv")
    let rawWelle6 = Rawtemplate.Load("files/raw/Welle6/templatePages.csv")
    let rawWelle5Count = rawWelle5.Rows |> Seq.length
    let rawWelle6Count = rawWelle6.Rows |> Seq.length

    let maxCount = rawWelle6.Rows |> Seq.maxBy (fun x -> x.Count)
    let minCount = rawWelle6.Rows |> Seq.minBy (fun x -> x.Count)
    ()

let rawBasic()=    
    let rawWelle5 = Rawbasic.Load("files/raw/Welle5/basicPages.json")
    let rawWelle6 = Rawbasic.Load("files/raw/Welle6/basicPages.json")
    let rawWelle5Count = rawWelle5.Length
    let rawWelle6Count = rawWelle6.Length

    let assignmentsWelle5 = rawWelle5 |> Seq.sumBy (fun x -> x.Templateguids.Length)
    let assignmentsWelle6 = rawWelle6 |> Seq.sumBy (fun x -> x.Templateguids.Length)
    ()
    
let plotTemplateCategoriesWelle6()=   
    let rawData = Rawtemplate.Load("files/raw/Welle6/templatePages.csv")

    let categorie1 = rawData.Rows |> Seq.map (fun x -> x.Category01)
    let categorie2 = rawData.Rows |> Seq.map (fun x -> x.Category02)
    let categorie3 = rawData.Rows |> Seq.map (fun x -> x.Category03)
    let categorie4 = rawData.Rows |> Seq.map (fun x -> x.Category04)
    let categorie5 = rawData.Rows |> Seq.map (fun x -> x.Category05)
    let categorie6 = rawData.Rows |> Seq.map (fun x -> x.Category06)
    let categorie7 = rawData.Rows |> Seq.map (fun x -> x.Category07)
    let categorie8 = rawData.Rows |> Seq.map (fun x -> x.Category08)
    let categorie9 = rawData.Rows |> Seq.map (fun x -> x.Category09)
    let categorie10 = rawData.Rows |> Seq.map (fun x -> x.Category10)
    let categorie11 = rawData.Rows |> Seq.map (fun x -> x.Category11)
    let categorie12 = rawData.Rows |> Seq.map (fun x -> x.Category12)
    let categorie13 = rawData.Rows |> Seq.map (fun x -> x.Category13)
    let categorie14 = rawData.Rows |> Seq.map (fun x -> x.Category14)
    let categorie15 = rawData.Rows |> Seq.map (fun x -> x.Category15)
    let categorie16 = rawData.Rows |> Seq.map (fun x -> x.Category16)
    let categorie17 = rawData.Rows |> Seq.map (fun x -> x.Category17)
    let categorie18 = rawData.Rows |> Seq.map (fun x -> x.Category18)
    let categorie19 = rawData.Rows |> Seq.map (fun x -> x.Category19)
    let categorieNotAvailable = rawData.Rows |> Seq.map (fun x -> x.Category_not_available)
    let layout = 
        Layout(            
            yaxis = Yaxis(``type`` = "log", autorange = true, title= "Anzahl der Elemente pro Schaltplan"),
            xaxis = Xaxis(tickangle=45.0),
            title = "Anzahl der Elemente pro Kategorie pro Schaltplan",
            showlegend = false
        )

    let plot =
        [
        Box(y=categorie1, name="Kategorie 1")
        Box(y=categorie2, name="Kategorie 2")
        Box(y=categorie3, name="Kategorie 3")
        Box(y=categorie4, name="Kategorie 4")
        Box(y=categorie5, name="Kategorie 5")
        Box(y=categorie6, name="Kategorie 6")
        Box(y=categorie7, name="Kategorie 7")
        Box(y=categorie8, name="Kategorie 8")
        Box(y=categorie9, name="Kategorie 9")
        Box(y=categorie10, name="Kategorie 10")
        Box(y=categorie11, name="Kategorie 11")
        Box(y=categorie12, name="Kategorie 12")
        Box(y=categorie13, name="Kategorie 13")
        Box(y=categorie14, name="Kategorie 14")
        Box(y=categorie15, name="Kategorie 15")
        Box(y=categorie16, name="Kategorie 16")
        Box(y=categorie17, name="Kategorie 17")
        Box(y=categorie18, name="Kategorie 18")
        Box(y=categorie19, name="Kategorie 19")
        Box(y=categorieNotAvailable, name="Nicht verfügbar")
        ]
        |> Chart.Plot
        |> Chart.WithHeight 400
        |> Chart.WithWidth 800
        |> Chart.WithLayout layout
    Chart.Show plot
    ()

[<EntryPoint>]
let main argv =
    //rawTemplate()
    //rawBasic()
    //plotTemplateCategoriesWelle6()

    //Exp1.countDataFilter()
    //Exp1.plotExperimentHistoryFilter()
    //Exp1.plotExperimentHistoryRanking()
    Exp1.plotFilterPredictions()
    Exp1.plotrankingPredictions()

    //Exp2.countDataRanking()

    //Exp2.plotExperimentHistory2()
    //Exp2.plotExperimentHistoryDifferentModells()

    //Exp2.plotExperimentHistory2Ranking()
    
    //Exp2.plotFilterPredictions()
    //Exp2.saveUniqueTestBasicGuids()
    //Exp2.countDataRanking()
    //Exp2.countDataFilter()
    Exp2.plotrankingPredictions()

    //ActivationFunctions.sigmoid()
    //ActivationFunctions.tanh()
    //ActivationFunctions.relu()    
    //ActivationFunctions.leakyrelu()
    printfn "done"
    System.Console.ReadKey() |> ignore
    0 // return an integer exit code
