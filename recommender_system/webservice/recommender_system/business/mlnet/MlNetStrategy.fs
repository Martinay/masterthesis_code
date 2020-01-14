module MlNetStrategy

open Microsoft.ML
open Microsoft.ML.Data

let preperationPipelinePath = "business/mlnet/preperationModel.zip"
let predictionModelRankingPath = "business/mlnet/modelRanking.onnx"
let predictionModelFilterPath = "business/mlnet/modelFilter.onnx"

let modelOutputLayerName  = "predictions_2/Identity:0"

[<CLIMutable>]
type PipelineInput ={
    [<ColumnName("matching")>] Matching: bool
    
    [<ColumnName("Basic_structure01")>] Basic_structure01: string
    [<ColumnName("Basic_structure02")>] Basic_structure02: string
    [<ColumnName("Basic_structure03")>] Basic_structure03: string
    [<ColumnName("Basic_structure04")>] Basic_structure04: string
    [<ColumnName("Basic_structure05")>] Basic_structure05: string
    [<ColumnName("Basic_structure06")>] Basic_structure06: string
    [<ColumnName("Basic_structure07")>] Basic_structure07: string
    [<ColumnName("Template_count")>] Template_count: float
    [<ColumnName("Template_structure01")>] Template_structure01: string
    [<ColumnName("Template_structure02")>] Template_structure02: string
    [<ColumnName("Template_structure03")>] Template_structure03: string
    [<ColumnName("Template_structure04")>] Template_structure04: string
    [<ColumnName("Template_structure05")>] Template_structure05: string
    [<ColumnName("Template_structure06")>] Template_structure06: string
    [<ColumnName("Template_structure07")>] Template_structure07: string
    [<ColumnName("Template_structure08")>] Template_structure08: string
    [<ColumnName("Template_category01")>] Template_category01: float
    [<ColumnName("Template_category02")>] Template_category02: float
    [<ColumnName("Template_category03")>] Template_category03: float
    [<ColumnName("Template_category04")>] Template_category04: float
    [<ColumnName("Template_category05")>] Template_category05: float
    [<ColumnName("Template_category06")>] Template_category06: float
    [<ColumnName("Template_category07")>] Template_category07: float
    [<ColumnName("Template_category08")>] Template_category08: float
    [<ColumnName("Template_category09")>] Template_category09: float
    [<ColumnName("Template_category10")>] Template_category10: float
    [<ColumnName("Template_category11")>] Template_category11: float
    [<ColumnName("Template_category12")>] Template_category12: float
    [<ColumnName("Template_category13")>] Template_category13: float
    [<ColumnName("Template_category14")>] Template_category14: float
    [<ColumnName("Template_category15")>] Template_category15: float
    [<ColumnName("Template_category16")>] Template_category16: float
    [<ColumnName("Template_category17")>] Template_category17: float
    [<ColumnName("Template_category18")>] Template_category18: float
    [<ColumnName("Template_category19")>] Template_category19: float
    [<ColumnName("Template_category_not_available")>] Template_category_not_available: float
    }

[<CLIMutable>]
type OnnxModelPipelineInput ={
        [<ColumnName("Feature_Template_Categories"); VectorType(20)>] Feature_Template_Categories: double[]
        [<ColumnName("Feature_Template_Count")>] Feature_Template_Count: float
        [<ColumnName("Feature_Basic_structure01"); VectorType(45)>] Feature_Basic_structure01: single[]
        [<ColumnName("Feature_Basic_structure02"); VectorType(45)>] Feature_Basic_structure02: single[]
        [<ColumnName("Feature_Basic_structure03"); VectorType(45)>] Feature_Basic_structure03: single[]
        [<ColumnName("Feature_Basic_structure05"); VectorType(43)>] Feature_Basic_structure05: single[]
        [<ColumnName("Feature_Template_structure01"); VectorType(45)>] Feature_Template_structure01: single[]
        [<ColumnName("Feature_Template_structure02"); VectorType(45)>] Feature_Template_structure02: single[]
        [<ColumnName("Feature_Template_structure03"); VectorType(45)>] Feature_Template_structure03: single[]
        [<ColumnName("Feature_Template_structure05"); VectorType(35)>] Feature_Template_structure05: single[]
        [<ColumnName("Feature_Template_structure06"); VectorType(45)>] Feature_Template_structure06: single[]
        [<ColumnName("Feature_Template_structure08"); VectorType(45)>] Feature_Template_structure08: single[]
    }

[<CLIMutable>]
type OnnxModelOutputFilter ={
    [<ColumnName("predictions/Identity:0")>] Prediction: single[]
}

[<CLIMutable>]
type OnnxModelOutputRanking ={
    [<ColumnName("predictions/Identity:0")>] Prediction: single[]
}

type OnnxModelOutput ={ Prediction: single[]}

let convertToPipelineInput (data:Models.PredictionData.Row) =   
    {
        Matching=false;
        
        Basic_structure01=data.Basic_structure01;
        Basic_structure02=data.Basic_structure02;
        Basic_structure03=data.Basic_structure03;
        Basic_structure04=data.Basic_structure04;
        Basic_structure05=data.Basic_structure05;
        Basic_structure06=data.Basic_structure06;
        Basic_structure07=data.Basic_structure07;
        
        Template_count=float data.Template_count;
        Template_structure01=data.Template_structure01;
        Template_structure02=data.Template_structure02;
        Template_structure03=data.Template_structure03;
        Template_structure04=data.Template_structure04;
        Template_structure05=data.Template_structure05;
        Template_structure06=data.Template_structure06;
        Template_structure07=data.Template_structure07;
        Template_structure08=data.Template_structure08;
        Template_category01=float data.Template_category01;
        Template_category02=float data.Template_category02;
        Template_category03=float data.Template_category03;
        Template_category04=float data.Template_category04;
        Template_category05=float data.Template_category05;
        Template_category06=float data.Template_category06;
        Template_category07=float data.Template_category07;
        Template_category08=float data.Template_category08;
        Template_category09=float data.Template_category09;
        Template_category10=float data.Template_category10;
        Template_category11=float data.Template_category11;
        Template_category12=float data.Template_category12;
        Template_category13=float data.Template_category13;
        Template_category14=float data.Template_category14;
        Template_category15=float data.Template_category15;
        Template_category16=float data.Template_category16;
        Template_category17=float data.Template_category17;
        Template_category18=float data.Template_category18;
        Template_category19=float data.Template_category19;
        Template_category_not_available=float data.Template_category_not_available;
        }
   

[<CLIMutable>]
type OnnxModelInput = OnnxModelPipelineInput

[<CLIMutable>]
type CustomMappingOutput =
    {
        [<ColumnName("inputStructureBasic01"); VectorType(45)>] mutable StructureBasic01: single[]        
        [<ColumnName("inputStructureBasic02"); VectorType(45)>] mutable StructureBasic02: single[]        
        [<ColumnName("inputStructureBasic03"); VectorType(45)>] mutable StructureBasic03: single[]        
        [<ColumnName("inputStructureBasic05"); VectorType(43)>] mutable StructureBasic05: single[] 
        [<ColumnName("inputStructureTemplate01"); VectorType(45)>] mutable StructureTemplate01: single[]       
        [<ColumnName("inputStructureTemplate02"); VectorType(45)>] mutable StructureTemplate02: single[]
        [<ColumnName("inputStructureTemplate03"); VectorType(45)>] mutable StructureTemplate03: single[]
        [<ColumnName("inputStructureTemplate05"); VectorType(35)>] mutable StructureTemplate05: single[]
        [<ColumnName("inputStructureTemplate06"); VectorType(45)>] mutable StructureTemplate06: single[]
        [<ColumnName("inputStructureTemplate08"); VectorType(45)>] mutable StructureTemplate08: single[]
        [<ColumnName("categories"); VectorType(20)>] mutable Categories: single[]
        [<ColumnName("templatecount"); VectorType(1)>] mutable TemplateCount: single[]
    }

let mlContext = MLContext()
let preperationPipelineModel, modelSchema = mlContext.Model.Load(preperationPipelinePath);
   
let customMapping (input:OnnxModelPipelineInput) (out:CustomMappingOutput) = 
    let categories = input.Feature_Template_Categories |> Seq.map single |> Seq.toArray
    let templateCount = single input.Feature_Template_Count
        
    out.Categories <- categories
    out.TemplateCount <- [|templateCount|]
    out.StructureBasic01 <- input.Feature_Basic_structure01
    out.StructureBasic02 <- input.Feature_Basic_structure02
    out.StructureBasic03 <- input.Feature_Basic_structure03
    out.StructureBasic05 <- input.Feature_Basic_structure05
    out.StructureTemplate01 <- input.Feature_Template_structure01
    out.StructureTemplate02 <- input.Feature_Template_structure02
    out.StructureTemplate03 <- input.Feature_Template_structure03
    out.StructureTemplate05 <- input.Feature_Template_structure05
    out.StructureTemplate06 <- input.Feature_Template_structure06
    out.StructureTemplate08 <- input.Feature_Template_structure08

let buildpredictionPipeline modelPath =     
    EstimatorChain()
        .Append(mlContext.Transforms.CustomMapping((fun i o -> customMapping i o), "MapFloatToSingle"))
        .Append(mlContext.Transforms.ApplyOnnxModel(modelFile = modelPath))
        //.Append(mlContext.Transforms.SelectColumns([|modelOutputLayerName|]))
        .Fit(mlContext.Data.LoadFromEnumerable(Array.empty<OnnxModelPipelineInput>))

let filterPredictionPipeline = 
    buildpredictionPipeline predictionModelFilterPath |> preperationPipelineModel.Append

let rankingPredictionPipeline = 
    buildpredictionPipeline predictionModelRankingPath |> preperationPipelineModel.Append
    
let predict (data:Models.PredictionData) (pipeline:ITransformer) getOutput = 
    let dataConverted = data.Rows |> Seq.map convertToPipelineInput
    
    let inputData = mlContext.Data.LoadFromEnumerable dataConverted
    let predictionData = pipeline.Transform(inputData)
    let wrappedData = getOutput predictionData

    let dataArray = wrappedData |> Seq.map (fun x-> x.Prediction |> Seq.head |> float) |> Seq.toList
    {Models.PredictionResult.Predictions=dataArray}

let predictFilter (data:Models.PredictionData) =    
    let getOutput data = mlContext.Data.CreateEnumerable<OnnxModelOutputFilter>(data, false, false) |> Seq.map (fun x-> {Prediction=x.Prediction})
    predict data filterPredictionPipeline getOutput

let predictRanking (data:Models.PredictionData) =    
    let getOutput data = mlContext.Data.CreateEnumerable<OnnxModelOutputRanking>(data, false, false) |> Seq.map (fun x-> {Prediction=x.Prediction})
    predict data rankingPredictionPipeline getOutput