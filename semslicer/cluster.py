from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd

def cluster(df):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(df['context'], show_progress_bar=True)
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(embeddings)
    df['cluster'] = kmeans.labels_
    return df

if __name__ == "__main__":
    # data_path = 'data/data/civil_comments_sampled.csv'
    # data_path = 'data/data/yahoo.csv'
    data_path = 'data/data/hotel.csv'
    # data_path = 'data/data/harmful_qa.csv'
    df = pd.read_csv(data_path)
    df = cluster(df)
    # sample a few examples from each cluster
    samples = df.groupby('cluster').apply(lambda x: x.sample(1))
    print(samples)
    df.to_csv(data_path, index=False)