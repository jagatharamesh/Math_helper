from dotenv import load_dotenv
from langchain_groq import ChatGroq

import os

load_dotenv()
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),model_name="llama-3.3-70b-versatile")
 
  
# we are not running the model locally,it is on groqcloud ,
# from there it is fetching the response
if __name__ == "__main__":
  response=llm.invoke("what are the different categories in maths")
  print(response.content)
