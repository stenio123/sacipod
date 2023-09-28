from tasks.preprocess_data import preprocess_data
from tasks.convert_vector_embeddings import convert_to_embeddings
from tasks.pinecone_create_index import upload_to_pinecone
from tasks.pinecone_semantic_query import search_podcast

# 1. Load JSON Data from a local file
with open('data/sample_transcription.json', 'r') as file:
    json_data = file.read()

# 2. Prepare data for search (cleans and adds sentiment analysis)
df = preprocess_data(json_data)
# print(df)

# 3. Convert to vector embeddings
df = convert_to_embeddings(df)
# print(df)

# 4. (Optional) Serialize data/ store S3 or HuggingFace
# TODO

# 5. Create index in Vector DB
#embedding_dimension = len(df['embeddings'].iloc[0])
#print(embedding_dimension)
#print(df['embeddings'].head())

index = upload_to_pinecone(df)

# 6. Execute query
search_query = "show me when Android was praised"
search_podcast(index, search_query)