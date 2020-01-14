module Models

open FSharp.Data

type DataBase = 
    | GL1 = 0
    | Welle5 = 1
    | Welle6 = 2

type BasicPage = JsonProvider<"files/basicPages_Sample.json">