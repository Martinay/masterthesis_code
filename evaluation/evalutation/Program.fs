// Learn more about F# at http://fsharp.org

open System
open FSharp.Data

type GuidList = JsonProvider<"files/testDataBasicPages/testBasicGuidsRanking.txt">

let filterBasicPages (basicPages:Models.BasicPage.Root[]) =
    if Config.Database = Models.DataBase.Welle6 
    then
        let ranking = GuidList.Load("files/testDataBasicPages/testBasicGuidsRanking.txt")
        let filter = GuidList.Load("files/testDataBasicPages/testBasicGuidsFilter.txt")
        let basicGuids = Seq.concat [ranking;filter] |> Seq.distinct |> Set.ofSeq
        basicPages |> Seq.where (fun x -> basicGuids.Contains x.SlotGuid)
    else 
        basicPages |> Seq.map id

[<EntryPoint>]
let main argv =
    printfn "%A" DateTime.Now
    let basicPages = Data.loadBasicPages
    printfn "loaded: %i" basicPages.Length
    let filteredBasicPages = filterBasicPages basicPages
    let basicPagesToEvaluate = filteredBasicPages |> Seq.take 500 |> Seq.toList
    printfn "evaluated: %i" basicPagesToEvaluate.Length
    let evaluation = Evaluate.evaluate basicPagesToEvaluate

    let text = sprintf "%A" evaluation
    printf "%s" text
    
    let filePath = sprintf "result%A.txt" Config.Database

    System.IO.File.WriteAllText(filePath, text)

    System.Console.ReadKey() |> ignore
    0 // return an integer exit code
