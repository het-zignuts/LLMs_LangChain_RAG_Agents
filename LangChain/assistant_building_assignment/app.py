import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

MODEL_NAME=os.getenv("MODEL")

user_input=""

file_path='prompt_template.txt'
with open(file_path, "r") as file:
    template=file.read()

prompt=PromptTemplate(input_variables=["topic"], template=template)
llm=ChatGroq(model=MODEL_NAME, temperature=0.1)
parser=StrOutputParser()

chain=prompt | llm | parser

while True:
    user_input=input("> User: Explain the topic: ")
    if user_input=="QUIT":
        break
    response=chain.invoke({"topic":user_input})
    print(f"AI says:\n{response}")