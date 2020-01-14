FROM mcr.microsoft.com/dotnet/core/aspnet:3.0-buster-slim AS base
WORKDIR /app
EXPOSE 80

RUN apt-get update \
	&& apt-get install -y python3 python3-pip \
	&& rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install pandas==0.25.2
RUN pip3 install tensorflow==2.0.0
RUN pip3 install keras==2.3.1

RUN pip3 install ptvsd==4.3.2
RUN pip3 install Flask==1.1.1

FROM mcr.microsoft.com/dotnet/core/sdk:3.0-buster AS build
WORKDIR /src
COPY ["recommender_system/recommender_system.fsproj", "recommender_system/"]
RUN dotnet restore "recommender_system/recommender_system.fsproj"
COPY . .
WORKDIR "/src/recommender_system"
RUN dotnet build "recommender_system.fsproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "recommender_system.fsproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "recommender_system.dll"]