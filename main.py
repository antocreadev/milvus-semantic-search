from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

client.create_collection(collection_name="test_colaaaasdqdalection", dimension=5)

print(client.list_collections())


