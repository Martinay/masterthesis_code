module Models

open FSharp.Data

type Strategy = 
    | LocalPython = 0
    | Azure = 1
    | Ibm = 2
    | MlNet = 3

type DataBase = WebRequests.DataBase

type BasicPage = WebRequests.BasicPage
    
type TemplatePagesCSV = CsvProvider<"business/data/files/sample/templatePages.csv">

type TemplatePage = TemplatePagesCSV.Row

type PreparedTemplates = 
    {Page:TemplatePage;
    Structure01_words:string;
    Structure02_words:string;
    Structure03_words:string;
    Structure04_words:string;
    Structure05_words:string;
    Structure06_words:string;
    Structure07_words:string;
    Structure08_words:string}

type PreparedBasicInfos = 
    {Page:BasicPage;
    Structure01_words:string;
    Structure02_words:string;
    Structure03_words:string;
    Structure04_words:string;
    Structure05_words:string;
    Structure06_words:string;
    Structure07_words:string}
    
type PredictionData = CsvProvider<"business/prediction_data_sample.csv",Separators="\t">

type PredictionResult = {Predictions:List<float>}