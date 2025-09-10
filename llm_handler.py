import ollama

client = ollama.Client()

model = "gpt-oss:20b"

prompt = "Write a Python function that calculates the factorial of a number."

response = client.generate(
    model=model,
    prompt=prompt
)

print("THE OLLLAMA RESPONSE:")
print(response.response)
