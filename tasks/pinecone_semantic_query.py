from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import pinecone
import settings

def search_podcast(index, query: str, top_k_results: int = 5, sentiment_threshold: float = 0.2):
    """
    Search podcasts based on the query and print the results.

    :param query: The search query.
    :param top_k_results: The number of top results to return.
    :param sentiment_threshold: The sentiment threshold for filtering results.
    """
    # Initialize Pinecone
    #pinecone.init(api_key=settings.PINECONE_API_KEY)

    # Load BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    # Tokenize and encode the query
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    xq = outputs['last_hidden_state'].mean(dim=1).detach().numpy().squeeze()

    # Query Pinecone index
    results = index.query(vector=xq.tolist(), top_k=top_k_results, include_metadata=True)
    
    print(f"Query Results: {results}")
    matches = results.get('matches')
    if not matches:
        print("No matches found.")
        return
    # Filter and sort based on sentiment
    filtered_results = []
    for match in matches:
        metadata = match['metadata']
        sentiment = metadata.get('sentiment', 0)  # Assuming sentiment is stored in metadata
        if sentiment > sentiment_threshold:  # Filter condition
            filtered_results.append(match)

    # Sort the results by sentiment
    sorted_results = sorted(filtered_results, key=lambda x: x.metadata['sentiment'], reverse=True)

    # Now process the sorted and filtered results
    for result in sorted_results[:top_k_results]:  # Limit to top results after sorting and filtering
        metadata = result.metadata
        print(f"ID: {result.id}, "
              f"Podcast Name: {metadata['podcast_name']}, "
              f"Speaker: {metadata['speaker']}, "
              f"Time: {metadata['time']}, "
              f"Sentiment: {metadata['sentiment']}, "
              f"Score: {result.score}")

