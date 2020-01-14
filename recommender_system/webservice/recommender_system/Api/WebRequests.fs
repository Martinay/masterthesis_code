module WebRequests

open System

type Recommendation = {TemplateGuid:Guid; Ranking:int}

type DataBase = 
    | GL1 = 0
    | Welle5 = 1
    | Welle6 = 2

type BasicPage = 
    {Structure01:string;
    Structure02:string; 
    Structure03:string;
    Structure04:string; 
    Structure05:string; 
    Structure06:string; 
    Structure07:string}

type RecommendWebRequest = {BasicInfos:BasicPage; Database:DataBase}
type RecommendWebRequestResponse = {Ranking:List<Recommendation>}