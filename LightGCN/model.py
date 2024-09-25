import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class LightGCNConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        from_, to_ = edge_index
        deg = torch.bincount(to_)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        return self.propagate(edge_index, x=x, norm=norm)

    def propagate(self, edge_index, x, norm):
        row, col = edge_index
        return torch.matmul(norm.view(-1, 1) * x[row], torch.ones(x.size(1)).to(x.device))

class RecSysGNN(nn.Module):
    def __init__(self, latent_dim, num_layers, num_users, num_items):
        super(RecSysGNN, self).__init__()
        self.embedding_user = nn.Embedding(num_users, latent_dim)
        self.embedding_item = nn.Embedding(num_items, latent_dim)
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])

    def forward(self, edge_index):
        user_emb = self.embedding_user(edge_index[0])
        item_emb = self.embedding_item(edge_index[1])

        for conv in self.convs:
            user_emb = conv(user_emb, edge_index)
            item_emb = conv(item_emb, edge_index)

        return user_emb, item_emb

def load_model(latent_dim, n_layers, n_users, n_items, model_path):
    model = RecSysGNN(latent_dim, n_layers, n_users, n_items)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    return model

def load_movies_data(file_path):
    return pd.read_csv(file_path)

def get_top_recommendations(user_id, selected_genres, model, movies_df, n_items, edge_index):
    with torch.no_grad():
        user_emb, item_emb = model(edge_index)

    user_embedding = user_emb[user_id]
    similarities = F.cosine_similarity(user_embedding, item_emb, dim=-1)
    top_indices = torch.argsort(similarities, descending=True)[:n_items]

    recommended_movie_ids = top_indices.cpu().numpy()
    recommendations = movies_df.iloc[recommended_movie_ids]
    
    recommendations['matching_genres'] = recommendations['genres'].apply(lambda x: sum(genre in x for genre in selected_genres))
    top_recommendations = recommendations.sort_values(by='matching_genres', ascending=False).head(10)

    return top_recommendations[['movieId', 'title']]

# Load movies data
movies_df = load_movies_data('ml-latest-small/movies.csv')

# Load ratings data to construct edge_index
ratings_df = pd.read_csv('ml-latest-small/ratings.csv')

# Create edge index based on user-item interactions
user_ids = ratings_df['userId'].unique()
item_ids = ratings_df['movieId'].unique()
n_users = len(user_ids)
n_items = len(item_ids)

# Mapping user and item IDs to indices
user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
item_to_index = {item_id: idx + n_users for idx, item_id in enumerate(item_ids)}  # Offset by n_users

# Create edge index
edge_index = torch.tensor([
    [user_to_index[user] for user in ratings_df['userId']],
    [item_to_index[item] for item in ratings_df['movieId']]
], dtype=torch.long)

# Example parameters
latent_dim = 64
n_layers = 3
model_path = 'lightgcn1_model.pth'

# Load the model
lightgcn = load_model(latent_dim, n_layers, n_users, n_items, model_path)

# Example function to get recommendations
def recommend(user_id, selected_genres, n_items=10):
    return get_top_recommendations(user_id, selected_genres, lightgcn, movies_df, n_items, edge_index)
