import pytest
from sacipod.tasks.pinecone_create_index import upload_to_pinecone
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Sample dataset for testing
test_data = pd.DataFrame({
    'sentence': ['sample text'],
    'podcast': ['podcast1'],
    'time_start': ['time1'],
    'time_end': ['time2'],
    'sentiment': ['sentiment1'],
    'embeddings': [np.array([1, 2, 3])]
})

@pytest.fixture
def pinecone_mocks():
    with patch('sacipod.tasks.pinecone_create_index.pinecone') as mock_pinecone:
        mock_pinecone.init = MagicMock()
        mock_pinecone.create_index = MagicMock()
        mock_pinecone.list_indexes = MagicMock(return_value=[])
        mock_pinecone.Index = MagicMock()
        yield mock_pinecone

def test_upload_to_pinecone(pinecone_mocks):
    upload_to_pinecone(test_data)
    pinecone_mocks.init.assert_called_once()
    pinecone_mocks.create_index.assert_called_once()
    pinecone_mocks.Index.assert_called_once()
    pinecone_mocks.Index().upsert.assert_called()

