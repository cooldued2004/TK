from openai import OpenAI
client = OpenAI()

prompt = "Write a short story about a robot learning emotions."

for temp in [0.2, 0.7, 1.2]:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    
    print(f"\n--- Temperature: {temp} ---")
    print(response.choices[0].message.content)