module Python.LocalPythonStrategy

open FSharp.Data

let endpoint = "http://host.docker.internal:8080"

type PythonResult = JsonProvider<"business/python/python_result_sample_json">

let predict route (data:Models.PredictionData) =    
    if not(data.Rows |> Seq.exists (fun _ -> true))
    then 
        {Models.PredictionResult.Predictions=[]}
    else
        let endpoint = sprintf "%s/%s" endpoint route
        let csvString = data.SaveToString('\t')
        let buffer = System.Text.Encoding.UTF8.GetBytes(csvString)
        let response = 
            Http.RequestString(
                endpoint, 
                httpMethod="POST",
                body=HttpRequestBody.BinaryUpload(buffer),
                headers = [ HttpRequestHeaders.ContentType(HttpContentTypes.Csv) ])
        
        let result = PythonResult.Parse(response) |> Seq.map float |> Seq.toList
        
        {Models.PredictionResult.Predictions=result}

let predictRanking (data:Models.PredictionData) =    
    predict "rank" data

let predictFilter (data:Models.PredictionData) =    
    predict "filter" data
