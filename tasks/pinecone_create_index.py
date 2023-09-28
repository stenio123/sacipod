import pinecone
import settings
import time

def upload_to_pinecone(df, index_name=settings.INDEX_NAME):
    PINECONE_API_KEY = settings.PINECONE_API_KEY
    PINECONE_ENV = settings.PINECONE_ENV

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

    pinecone_data = [
        {
            'id': str(idx + 1),  # id
            'values': row['embeddings'].tolist(),  # values
            'metadata': {  # metadata
                'text': row['sentence'][:settings.MAX_TEXT_LENGTH],
                'podcast_name': row['podcast'],
                'speaker': '',  # if speaker info is not available in df
                'time': f"{row['time_start']} - {row['time_end']}",
                'sentiment': row['sentiment']
            }
        }
        for idx, row in df.iterrows()
    ]

    # Create index
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=768, metric='cosine')
        time.sleep(5)

    # Upsert data to Pinecone index
    index = pinecone.Index(index_name=index_name)
    for batch in iter_batches(pinecone_data, batch_size=25):
        try:
            index.upsert(vectors=batch)
        except ValueError as e:
            print(f"Error upserting batch: {e}")
            break  # Exit the loop to prevent further errors
    return index

def iter_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
