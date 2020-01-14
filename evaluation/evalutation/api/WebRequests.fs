module WebRequests

open System
open System.Text.Json
open System.Text.Json.Serialization
open FSharp.Data

let jsonOptions = JsonSerializerOptions()
jsonOptions.Converters.Add(JsonFSharpConverter())

type BasicPage = 
    {Structure01:string;
    Structure02:string; 
    Structure03:string;
    Structure04:string; 
    Structure05:string; 
    Structure06:string; 
    Structure07:string}
    
type Recommendation = {TemplateGuid:Guid; Ranking:int}

type RecommendWebRequest = {BasicInfos:BasicPage; Database:Models.DataBase}
type RecommendWebRequestResponse = {Ranking:List<Recommendation>}

let buildWebRequest (basicPage:Models.BasicPage.Root) =
    let structure01 = defaultArg basicPage.Structure01 ""
    let structure02 = defaultArg basicPage.Structure02 ""
    let structure03 = defaultArg basicPage.Structure03 ""
    let structure04 = defaultArg basicPage.Structure04 ""
    let structure05 = defaultArg basicPage.Structure05 ""
    let structure06 = defaultArg basicPage.Structure06 ""
    let structure07 = defaultArg basicPage.Structure07 ""
    let basicInfo = {Structure01=structure01;Structure02=structure02;Structure03=structure03;Structure04=structure04;Structure05=structure05;Structure06=structure06;Structure07=structure07}

    let request = {BasicInfos=basicInfo; Database=Config.Database}
    let json = JsonSerializer.Serialize(request, jsonOptions)

    let executeRequest() = 
        Http.RequestString(
            Config.Endpoint, 
            httpMethod="POST",
            body=HttpRequestBody.TextRequest(json),
            headers = [ HttpRequestHeaders.ContentType(HttpContentTypes.Json) ])
    executeRequest

let deserializeResponse (responseString:string) =
    let response = JsonSerializer.Deserialize<RecommendWebRequestResponse>(responseString, jsonOptions)
    response.Ranking
