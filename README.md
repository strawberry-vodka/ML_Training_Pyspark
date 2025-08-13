import asyncio
import nest_asyncio
import time
nest_asyncio.apply()

from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="")

semaphore = asyncio.Semaphore(60)  

async def summarize_text(text):
    async with semaphore:
        response = await client.responses.create(
            model="gpt-4o-mini",
            instructions="You are a helpful AI assistant working for a travel booking company - CheapOAir. Summarize the following call transcript in 150-200 words, capturing all key issues. If the customer mentions a transaction number, booking number, or other key PII-masked details, include them." , 
            input=text
        )
        return response.output_text

async def process_all(df, text_column):
    batch_size = 60
    count = 0
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        tasks = [summarize_text(t) for t in batch[text_column]]
        results = await asyncio.gather(*tasks)
        df.loc[batch.index, "masked_summary"] = results
        print("done")
        count += 1
        if count>20:
            time.sleep(15)
            count = 0
            print("Timer executed")
    return df
