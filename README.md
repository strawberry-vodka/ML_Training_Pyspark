from openai import OpenAI

client = OpenAI()

def classify_persona(transcript):
    prompt = f"""
    You are an assistant that tags travel personas based on call transcripts.
    The categories are:
    - Group type: Solo, Family, Couple, Friends
    - Travel purpose: Business, Leisure, Mixed

    Transcript:
    "{transcript}"

    Return output strictly in JSON format:
    {{
        "group_type": "...",
        "travel_purpose": "..."
    }}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a classification assistant."},
                  {"role": "user", "content": prompt}]
    )

    return resp.choices[0].message.content

# Example
print(classify_persona("I want to book a flight for me and my two kids for summer vacation in Spain."))
