from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, MilvusClient
import spacy
import numpy as np

nlp = spacy.load("fr_core_news_md")

connections.connect("default", uri="http://localhost:19530", user="root", password="Milvus")
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

name = "ok"

if not client.has_collection(name):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),  
        FieldSchema(name="word", dtype=DataType.VARCHAR, max_length=255),  
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=300)  
    ]
    schema = CollectionSchema(fields, description="Collection de test pour recherche de similarité de mots")
    client.create_collection(collection_name=name, schema=schema)
print("Liste des collections :")
print(client.list_collections())

print("\n")


print("Description de la collection :")
print(client.describe_collection(name))


print("\n")


collection = Collection(name=name)  

words = ["chat", "chien", "poisson", "oiseau", "souris", "pomme", "banane", "orange", "poire", "fraise"]

print(f"Liste des mots disponibles : {words}")
print("\n")

ids = np.arange(len(words))
embeddings = np.array([nlp(word).vector for word in words])

collection.insert([ids.tolist(), words, embeddings.tolist()]) 

collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
#  IVF_FLAT : Indexation vectorielle par approximation (IVF) avec index plat : recherche rapide mais précision moindre
#  L2 : Distance euclidienne
#  nlist : Nombre de cellules dans l'index : plus il y a de cellules, plus la recherche est rapide mais la précision est moindre

collection.load()

query = "framboise"
query_embedding = nlp(query).vector

search_result = collection.search(
    [query_embedding],  
    "embedding",      
    param={"metric_type": "L2", "top_k": 2}, 
    limit=2  
)
# L2 : Distance euclidienne
# top_k : Nombre de résultats à retourner
print("-------------")
print("\n")
print(f"Résultat verbose :")
print(search_result)
print("\n")
result_str = str(search_result[0]).split(",")[0][-1]
result_int = int(result_str)
print(f"Le mot le plus similaire à {query} est : {words[result_int]}")
print("\n")

while True : 
    # On demande à l'utilisateur de rentrer un mot et on lui retourne le mot le plus similaire

    print(f"Liste des mots disponibles : {words}")
    print("\n")
    query = input("Entrez un mot : ")
    query_embedding = nlp(query).vector

    search_result = collection.search(
        [query_embedding],  
        "embedding",      
        param={"metric_type": "L2", "top_k": 2}, 
        limit=2  
    )
    print("-------------")
    print("\n")
    print(f"Résultat verbose :")
    print(search_result)
    print("\n")
    result_str = str(search_result[0]).split(",")[0][-1]
    result_int = int(result_str)
    print(f"Le mot le plus similaire à {query} est : {words[result_int]}")
    print("\n")
