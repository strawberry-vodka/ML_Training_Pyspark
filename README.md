import asyncio
import pandas as pd
from openai import AsyncOpenAI

# Async OpenAI client
client = AsyncOpenAI(api_key="YOUR_API_KEY")

# Async summarization function
async def summarize_text_async(text):
    try:
        response = await client.responses.create(
            model="gpt-4o",
            instructions=(
                "You are a helpful AI assistant working for a travel booking company - CheapOAir "
                "that offers users to make their travel bookings. You are given a full transcript of "
                "a phone call between a customer and a support agent. The customer may call for a "
                "variety of reasons, and a single call can include multiple topics. Your job is to "
                "summarize the issue(s) mentioned in the entire transcript in 150-200 words. "
                "If the customer mentions a transaction number, booking number, or other key details "
                "like PII, include this information in the summarized text."
            ),
            input=text
        )
        return response.output_text
    except Exception as e:
        return f"ERROR: {str(e)}"

# Process a batch of transcripts concurrently
async def process_batch(df_batch, col_name):
    tasks = [
        summarize_text_async(row[col_name])
        for _, row in df_batch.iterrows()
    ]
    return await asyncio.gather(*tasks)

# Main batch processor
def run_all_batches(df, col_name, batch_size=250):
    all_summaries = []
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        summaries = asyncio.run(process_batch(batch_df, col_name))
        all_summaries.extend(summaries)
        print(f"Processed {len(all_summaries)} / {len(df)} transcripts")
    return all_summaries

# Example usage
if __name__ == "__main__":
    # Example dataframe with masked transcripts
    calls_df = pd.DataFrame({
        "masked_text": [
            "Customer called regarding booking ID 12345, had issues with flight rescheduling...",
            "User reported payment failure on transaction ID TXN-98765 and wanted refund...",
            # ... 4,400 transcripts total
        ]
    })

    # Run summarization
    calls_df["summary"] = run_all_batches(calls_df, "masked_text")

    # Final check
    print(calls_df.head())
