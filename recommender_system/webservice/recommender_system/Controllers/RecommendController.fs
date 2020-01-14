namespace recommender_system.Controllers

open System
open Microsoft.AspNetCore.Mvc
open Microsoft.Extensions.Logging

[<Route("api/[controller]")>]
[<ApiController>]
type RecommendController (logger : ILogger<RecommendController>) =
   inherit ControllerBase()
       
    [<HttpPost>]
    member _.Post([<FromBody>] request:WebRequests.RecommendWebRequest) =
        let templates = Data.loadTemplatePages(request.Database)
        let basicInfos = Data.prepareBasicPage request.BasicInfos
        let filterstrategy, rankingStrategy = StrategyHelper.getFilterAndRankingFunction Models.Strategy.LocalPython

        let filteredTemplatePages = Filter.filter filterstrategy basicInfos templates

        let ranking = Rank.rank rankingStrategy basicInfos filteredTemplatePages

        let response = {WebRequests.RecommendWebRequestResponse.Ranking=ranking}
        ActionResult<WebRequests.RecommendWebRequestResponse>(response)
