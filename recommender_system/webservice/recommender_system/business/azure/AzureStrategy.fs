module AzureStrategy

open FSharp.Data

let endpointFilter = "<enter endpoint>"
let tokenFilter = "<enter token>"
let endpointRanking = "<enter endpoint>"
let tokenRanking = "<enter token>"

type PredictionResult = JsonProvider<"business/azure/prediction_response_sample_json">

let predictionRequestBody values = sprintf "{\"data\": [%s]}" values
let predictionRequestBodyValue = sprintf "[\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"]"

let predict (data:Models.PredictionData) endpoint token = 
    let convertToRequest (line:Models.PredictionData.Row) =
        let valueString = predictionRequestBodyValue line.Basic_structure01 line.Basic_structure02 line.Basic_structure03 line.Basic_structure04 line.Basic_structure05 line.Basic_structure06 line.Basic_structure07 line.Basic_structure01_words line.Basic_structure02_words line.Basic_structure03_words line.Basic_structure04_words line.Basic_structure05_words line.Basic_structure06_words line.Basic_structure07_words line.Template_count line.Template_category01 line.Template_category02 line.Template_category03 line.Template_category04 line.Template_category05 line.Template_category06 line.Template_category07 line.Template_category08 line.Template_category09 line.Template_category10 line.Template_category11 line.Template_category12 line.Template_category13 line.Template_category14 line.Template_category15 line.Template_category16 line.Template_category17 line.Template_category18 line.Template_category19 line.Template_category_not_available line.Template_structure01 line.Template_structure02 line.Template_structure03 line.Template_structure04 line.Template_structure05 line.Template_structure06 line.Template_structure07 line.Template_structure01_words line.Template_structure02_words line.Template_structure03_words line.Template_structure04_words line.Template_structure05_words line.Template_structure06_words line.Template_structure07_words
        valueString
    let valueStrings = data.Rows |> Seq.take 20 |> Seq.map convertToRequest
    let valueString = System.String.Join(',', valueStrings)
    let request = predictionRequestBody valueString
    let response = 
        Http.RequestString(
            endpoint, 
            httpMethod="POST",
            body=HttpRequestBody.TextRequest(request),
            headers = 
                [ 
                HttpRequestHeaders.ContentType(HttpContentTypes.Json); 
                HttpRequestHeaders.Accept(HttpContentTypes.Json);
                HttpRequestHeaders.Authorization(sprintf "Bearer %s" token);
                ])
    let responseWithoutStart = response.Remove(0,14)
    let responseTrimmed = responseWithoutStart.Remove(responseWithoutStart.Length - 2,2)

    let result = PredictionResult.Parse(responseTrimmed) |> Seq.map float |> Seq.toList
    
    {Models.PredictionResult.Predictions=result}

let predictFilter (data:Models.PredictionData) =    
    predict data endpointFilter tokenFilter

let predictRanking (data:Models.PredictionData) =    
    predict data endpointRanking tokenRanking