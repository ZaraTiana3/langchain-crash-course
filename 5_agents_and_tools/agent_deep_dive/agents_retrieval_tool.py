from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import (AgentExecutor, create_structured_chat_agent)
from langchain.memory import ConversationBufferMemory
import os

#This could be an alternative to agents_createdocstore because we have access to retrieval 
#We can use ConversationBufferMemory so everything is okay. They act the same way but this is shorter.
#https://python.langchain.com/v0.2/docs/tutorials/qa_chat_history/#retrieval-tool

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key = "AIzaSyBixXBp6k1wDIG1VHOrw3SMygX8Fn7xg3w")
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join("D:\ongoing\python_project\LangChain", "rag1","db", "chroma_db_with_metadata")
print("Persistent_directory: ", persistent_directory)
db = Chroma(persist_directory=persistent_directory,embedding_function=embeddings)
retriever = db.as_retriever(
    search_type= "similarity_score_threshold",
    search_kwargs = {"k": 3, "score_threshold": 0.4}
)

tool = create_retriever_tool(
    retriever,
    "SQL_retriever",
    "Gives results about the story of Odysseus and Romeo and Juliet life",
)

tools = [tool]

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-001",  api_key="AIzaSyBixXBp6k1wDIG1VHOrw3SMygX8Fn7xg3w")

prompt=hub.pull("hwchase17/structured-chat-agent")

agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# AgentExecutor is responsible for managing the interaction between the user input, the agent, and the tools
# It also handles memory to ensure context is maintained throughout the conversation
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,  # Use the conversation memory to maintain context
    handle_parsing_errors=True,  # Handle any parsing errors gracefully
)

initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])

     # Add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
