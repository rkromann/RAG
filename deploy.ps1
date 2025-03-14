az login
az group create -n rg-RAG -l westeurope
az acr create -n acrrag763457 -g rg-RAG -l westeurope --sku Basic
az acr update -n acrrag763457 --admin-enabled true
az acr credential show --name acrrag763457.azurecr.io
docker login acrrag763457.azurecr.io
docker tag streamlit acrrag763457.azurecr.io/streamlit:v1
docker push acrrag763457.azurecr.io/streamlit:v1

az appservice plan create -g rg-RAG -l westeurope -n plan-RAG --is-linux --sku B3

az webapp create -n ws-RAG -p plan-RAG -g rg-RAG -i acrrag763457.azurecr.io/streamlit:v1 -s acrrag763457 -w aWoR74LBENDmdPV/dA52aulwG2eDjMGeDgfwa64MVD+ACRDxYDlw