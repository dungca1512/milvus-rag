from pymilvus import connections

connections.connect(default=True, host="localhost", port="19530")

print("Connected to Milvus server") 