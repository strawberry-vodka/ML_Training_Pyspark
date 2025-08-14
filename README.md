import pandas as pd
from openai import OpenAI
from sklearn.cluster import KMeans
import numpy as np

client = OpenAI(api_key="YOUR_API_KEY")

# 1. Load DataFrame with transcripts
df = pd.read_csv("transcripts.csv")  # assuming `transcript` column

# 2. Generate embeddings
embeddings = []
for text in df['transcript']:
    resp = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embeddings.append(resp.data[0].embedding)

embeddings = np.array(embeddings)

# 3. Cluster transcripts
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings)

# 4. Summarize per cluster
generic_prompts = []
for cluster_id in df['cluster'].unique():
    sample_texts = df[df['cluster'] == cluster_id]['transcript'].sample(5, random_state=42)
    combined_text = "\n---\n".join(sample_texts)

    prompt = f"""
    The following are customer service call transcripts from the same topic category.
    Summarize the common purpose, user intents, and recommended agent responses.
    Output in a concise form.
    {combined_text}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a call center analysis assistant."},
                  {"role": "user", "content": prompt}]
    )

    generic_prompts.append(resp.choices[0].message.content)

# 5. Now convert these summaries into system prompts
system_prompts = []
for desc in generic_prompts:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a system prompt generator."},
                  {"role": "user", "content": f"Convert the following intent description into a reusable system prompt:\n{desc}"}]
    )
    system_prompts.append(resp.choices[0].message.content)

df_prompts = pd.DataFrame({"cluster": range(len(system_prompts)), "system_prompt": system_prompts})
print(df_prompts)
