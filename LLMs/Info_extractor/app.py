import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY=os.getenv("INFO_EXTRACTOR_API_KEY")
MODEL_NAME=os.getenv("MODEL")

user_query=input("Enter your query here: ")

system_prompt="""
You are an information extrator.

Output: 
- You should strictly output JSON only.
- Output Schema:
    - name : String
    - email: Email-String
    - issue_summary: String
    - urgency: Enum(high, medium, low)

Constraints:
- Strictly give response in JSON format adhering to output schema.
- Do not give any additional text or explanation in response
- Response must contain all the specified fields of schema.
- If a detail is not extracted, keep the value of the field as empty string
- Issue summary should be accurate and concise.
- Set the urgency field value from "high", "low" and "medium" only. 
- Only JSON of Output Schema should be there in response irrespective of message.

"""

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
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_query
        }
    ],
    "temperature": 0.1,
    "max_tokens": 200

}
try:
    response=requests.post(url, json=payload, headers=headers)
    print(response.json()["choices"][0]["message"]["content"])
except Exception as e:
    print("Raised an exception: " , e)