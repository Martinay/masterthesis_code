module Data

let splitIntoWords (text:string)=
    text |> Seq.chunkBySize 3 |> Seq.map System.String |> String.concat " "

let prepareBasicPage (basicInfos:Models.BasicPage)=
        {Models.PreparedBasicInfos.Page=basicInfos;
        Models.PreparedBasicInfos.Structure01_words= splitIntoWords basicInfos.Structure01;
        Models.PreparedBasicInfos.Structure02_words= splitIntoWords basicInfos.Structure02;
        Models.PreparedBasicInfos.Structure03_words= splitIntoWords basicInfos.Structure03;
        Models.PreparedBasicInfos.Structure04_words= splitIntoWords basicInfos.Structure04;
        Models.PreparedBasicInfos.Structure05_words= splitIntoWords basicInfos.Structure05;
        Models.PreparedBasicInfos.Structure06_words= splitIntoWords basicInfos.Structure06;
        Models.PreparedBasicInfos.Structure07_words= splitIntoWords basicInfos.Structure07;}

let templateCache =
    let prepareTemplates (row:Models.TemplatePage) =            
        {Models.PreparedTemplates.Page=row;
        Models.PreparedTemplates.Structure01_words= splitIntoWords row.Structure01;
        Models.PreparedTemplates.Structure02_words= splitIntoWords row.Structure02;
        Models.PreparedTemplates.Structure03_words= splitIntoWords row.Structure03;
        Models.PreparedTemplates.Structure04_words= splitIntoWords row.Structure04;
        Models.PreparedTemplates.Structure05_words= splitIntoWords row.Structure05;
        Models.PreparedTemplates.Structure06_words= splitIntoWords row.Structure06;
        Models.PreparedTemplates.Structure07_words= splitIntoWords row.Structure07;
        Models.PreparedTemplates.Structure08_words= splitIntoWords row.Structure08;}
    let loadData(filePath:string) =
        printfn "loading %s" filePath
        let csvData = Models.TemplatePagesCSV.Load(filePath)
        csvData.Rows |> Seq.map prepareTemplates |> Seq.toList

    dict[ Models.DataBase.GL1, lazy (loadData "business/data/files/gl1/templatePages.csv");
    Models.DataBase.Welle5, lazy (loadData "business/data/files/welle5/templatePages.csv");
    Models.DataBase.Welle6, lazy (loadData "business/data/files/welle6/templatePages.csv")
    ]

let loadTemplatePages (database:Models.DataBase) =    
    let templates = ``|Lazy|`` templateCache.[database]
    templates