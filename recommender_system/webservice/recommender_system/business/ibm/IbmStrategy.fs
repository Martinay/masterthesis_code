module IbmStrategy

open FSharp.Data

let endpointToken = "https://iam.cloud.ibm.com/identity/token"
let endpointFilter = "<enter endpoint>"

type PredictionResult = JsonProvider<"business/ibm/ibm_result_sample_json">
type TokenResult = JsonProvider<"business/ibm/ibm_token_result_sample_json">

let mutable accessToken = "<enter access token>"

let predictionRequestBody values = sprintf "{\"input_data\": [{\"fields\": [\"Basic_structure01\",\"Basic_structure02\",\"Basic_structure03\",\"Basic_structure04\",\"Basic_structure05\",\"Basic_structure06\",\"Basic_structure07\",\"Basic_structure01_words\",\"Basic_structure02_words\",\"Basic_structure03_words\",\"Basic_structure04_words\",\"Basic_structure05_words\",\"Basic_structure06_words\",\"Basic_structure07_words\",\"Template_count\",\"Template_structure01\",\"Template_structure02\",\"Template_structure03\",\"Template_structure04\",\"Template_structure05\",\"Template_structure06\",\"Template_structure07\",\"Template_category01\",\"Template_category02\",\"Template_category03\",\"Template_category04\",\"Template_category05\",\"Template_category06\",\"Template_category07\",\"Template_category08\",\"Template_category09\",\"Template_category10\",\"Template_category11\",\"Template_category12\",\"Template_category13\",\"Template_category14\",\"Template_category15\",\"Template_category16\",\"Template_category17\",\"Template_category18\",\"Template_category19\",\"Template_category_not_available\",\"Template_structure01_words\",\"Template_structure02_words\",\"Template_structure03_words\",\"Template_structure04_words\",\"Template_structure05_words\",\"Template_structure06_words\",\"Template_structure07_words\"],\"values\": [%s]}]}" values
let predictionRequestBodyValue = sprintf "[\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",%i,\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"]"

let renewToken =
    let response = 
        Http.RequestString(
            endpointToken, 
            httpMethod="POST",
            body=HttpRequestBody.FormValues["grant_type", "urn:ibm:params:oauth:grant-type:apikey";"apikey", "<enter apikey>"],
            headers = 
                [ 
                HttpRequestHeaders.ContentType(HttpContentTypes.FormValues);
                ])

    let tokenResponse = TokenResult.Parse(response)
    accessToken <- tokenResponse.AccessToken
    ()

let predict (data:Models.PredictionData) endpoint = 
    let convertToRequest (line:Models.PredictionData.Row) =
        let valueString = predictionRequestBodyValue line.Basic_structure01 line.Basic_structure02 line.Basic_structure03 line.Basic_structure04 line.Basic_structure05 line.Basic_structure06 line.Basic_structure07 line.Basic_structure01_words line.Basic_structure02_words line.Basic_structure03_words line.Basic_structure04_words line.Basic_structure05_words line.Basic_structure06_words line.Basic_structure07_words line.Template_count line.Template_structure01 line.Template_structure02 line.Template_structure03 line.Template_structure04 line.Template_structure05 line.Template_structure06 line.Template_structure07  line.Template_category01 line.Template_category02 line.Template_category03 line.Template_category04 line.Template_category05 line.Template_category06 line.Template_category07 line.Template_category08 line.Template_category09 line.Template_category10 line.Template_category11 line.Template_category12 line.Template_category13 line.Template_category14 line.Template_category15 line.Template_category16 line.Template_category17 line.Template_category18 line.Template_category19 line.Template_category_not_available line.Template_structure01_words line.Template_structure02_words line.Template_structure03_words line.Template_structure04_words line.Template_structure05_words line.Template_structure06_words line.Template_structure07_words
        valueString
    let valueStrings = data.Rows |> Seq.take 10 |> Seq.map convertToRequest
    let valueString = System.String.Join(',', valueStrings)
    let request = predictionRequestBody valueString
    let response = 
        Http.Request(
            endpoint, 
            httpMethod="POST",
            body=HttpRequestBody.TextRequest(request),
            headers = 
                [ 
                HttpRequestHeaders.ContentType(HttpContentTypes.Json); 
                HttpRequestHeaders.Accept(HttpContentTypes.Json);
                "ML-Instance-ID", "<enter instance id>"
                HttpRequestHeaders.Authorization("Bearer <enter bearer token>");
                ])
    let a = response.Body
    
    //let result = PythonResult.Parse(response) |> Seq.map float |> Seq.toList
    
    {Models.PredictionResult.Predictions=[]}

let predictFilter (data:Models.PredictionData) =    
    predict data endpointFilter

let predictRanking (data:Models.PredictionData) =    
    {Models.PredictionResult.Predictions=[(new System.Random()).Next(1, 1000) |> float]}