from transformers import AutoModel, AutoTokenizer

def convert_to_embeddings(df):
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    embeddings = []

    for index, row in df.iterrows():
        inputs = tokenizer(row['sentence'], return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        # Using .squeeze() to remove single-dimensional entries from the shape of the numpy array
        emb = outputs['last_hidden_state'].mean(dim=1).detach().numpy().squeeze()
        embeddings.append(emb)

    df['embeddings'] = embeddings
    return df

