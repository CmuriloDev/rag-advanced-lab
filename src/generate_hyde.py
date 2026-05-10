from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

def generate_hyde(query):
    prompt = f"""
Convert the following colloquial medical complaint
into a technical medical description:

Complaint:
{query}

Technical description:
"""

    output = generator(
        prompt,
        max_new_tokens=40,
        do_sample=True
    )

    generated = output[0]["generated_text"]

    return generated