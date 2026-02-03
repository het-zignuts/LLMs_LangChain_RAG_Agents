import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY=os.getenv("INFO_EXTRACTOR_API_KEY")
MODEL_NAME=os.getenv("MODEL")

file_path='prompt_template.txt'

with open(file_path, "r") as file:
    template=file.read()

SYSTEM_INSTRUCTIONS="""
- You are a text extraction and trasformation engine

CONSTRAINTS:
- Stricly generate response in either JSON or YAML format as asked by the user.
- Do not generate any additional text, explanation or communicative sentences.
- Stricly keep the fields in response object as per the output schema and type asked.
- Be grounded to the context strictly, if provided.
- Do not hallucinate much and keep the responses concise.
"""

user_query=input("Enter your query here: ")
context=input("Please provide context to get enhanced results: ")
output_fmt=input("Enter the desired output format: ")
output_schema=input("Enter output fields (name: type): ")
examples=input("Enter a few examples (Input:..., Output: ....): ")

user_prompt=template.format(
    user_query=user_query,
    OUTPUT_TYPE=output_fmt,
    OUTPUT_SCHEMA=output_schema,
    EXAMPLES= examples
)

url="https://api.groq.com/openai/v1/chat/completions"

headers={
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload={
    "model": MODEL_NAME,
    "messages":[
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTIONS
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],
    "temperature": 0.1,
    "max_tokens": 300

}
try:
    response=requests.post(url, json=payload, headers=headers)
    print(response.json()["choices"][0]["message"]["content"])
except Exception as e:
    print("Raised an exception: " , e)