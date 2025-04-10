@echo off
echo Starting docker-compose services...
docker-compose up -d

echo Starting Milvus container...
docker run -p 8000:3000 -e MILVUS_URL=192.168.10.64:19530 zilliz/attu:v2.4

pause
