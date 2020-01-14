module Data

let loadBasicPages = 
    let filePath = 
        match Config.Database with
        | Models.DataBase.GL1 -> "files/basicPages_GL1.json"
        | Models.DataBase.Welle5 -> "files/basicPages_Welle5.json"
        | Models.DataBase.Welle6 -> "files/basicPages_Welle6.json"

    let pages = Models.BasicPage.Load(filePath)
    pages