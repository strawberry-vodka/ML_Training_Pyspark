def summarize_text(text):
    response = client.responses.create(
        model="gpt-4o",
        instructions="You are a helpful AI assistant working for a travel booking company - CheapOAir that offers user make their travel booking. You are given a full transcript of a phone call between a customer and a support agent. The customer may call for a variety of reasons, and a single call can include multiple topics. Your job is to summarize the issue(s) mentioned in the entire transcript in 150-200 words. In case, the customer mentions a transcation number/booking number or other key details like PII, your job is to include the information on the summarized text.",
        input=text
    )
    
    #print(response.output_text)
    print("Done")
    return response.output_text
