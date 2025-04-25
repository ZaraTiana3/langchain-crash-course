from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain.chains import LLMChain, SequentialChain


#When making sequential chain, actually we still use PromptTemplate with LLMChain
model =  ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-001", api_key="AIzaSyBHLW9V4R8MaQP31f4VAjx-7GvZPVs4-e4")

template = "I want to open a  {cuisine} restaurant. Suggest only one fancy name for that without any meaning"
prompt_template = PromptTemplate(
    input_variables=['cuisine'],
    template= template)
name_chain = LLMChain(llm=model, prompt = prompt_template, output_key='restaurant')

template= "Suggest some items for {restaurant}. And when presenting the result. I want you to say: This is some items for {restaurant}  "
restaurant_template = PromptTemplate(
    input_variables= ['restaurant'],
    template = template)
rest_chain = LLMChain(llm=model, prompt = restaurant_template, output_key='menu_items')

chain = SequentialChain(chains=[name_chain, rest_chain], input_variables=['cuisine'], output_variables=['cuisine','menu_items'])

result = chain({"cuisine": "Mexico"})
print(result)




