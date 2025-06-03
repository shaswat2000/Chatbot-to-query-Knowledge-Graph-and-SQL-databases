
import streamlit as st
import sys
import io

import os
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain.chains import GraphCypherQAChain

from langchain.tools import Tool
from langchain.agents import initialize_agent
import re
from sqlalchemy import text
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains.sql_database.prompt import PROMPT
from langchain.prompts import PromptTemplate

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

db_global = SQLDatabase.from_uri("***")

def query_sql_database(question: str) -> str:
    db = SQLDatabase.from_uri("***")

    query_chain = create_sql_query_chain(llm, db)

    promptsql = """
You are an expert SQL agent. You only write PostgreSQL queries using SQLAlchemy-compatible syntax.
Never use double double-quotes (""). Use single double-quotes for identifiers (columns, tables), and single quotes for string or datetime values.

You are working with the following tables:
- occupancy_sensors_week(Columns: timestamp, room, occupancy)
- temperature_sensors_week(Columns: timestamp, room, temperature)

Examples:
Q: Show all rooms with temperature above 24
A: SELECT "room", "temperature" FROM "temperature_sensors_week" WHERE "temperature" > 24;

Q: On 4th May 2025, which rooms had temperature above 23?
A: SELECT DISTINCT "room" FROM "temperature_sensors_week" WHERE "temperature" > 23 AND "timestamp" >= '2025-05-04 00:00:00' AND "timestamp" < '2025-05-05 00:00:00';

"""
    # 4. Ask a natural language question
    response = query_chain.invoke({
        "question": promptsql + f"\nQ: {question}\nA:",
    })

    match = re.search(r"SQLQuery:\s*(.*)", response)
    if match:
        sql = match.group(1)
        print("Extracted SQL:", sql)
    else:
        print("No SQL query found.")

    with db._engine.connect() as conn:
        result = conn.execute(text(sql)).fetchall()
        print("Result:", result)

    prompt = f"""Question: {question}
    SQL: {sql}
    Result: {result[:10]}

    Can you express the result of the sql query in natural language?"""

    return llm.invoke(prompt)


def query_knowledge_graph(question: str) -> str:
    ## Graphdb configuration
    NEO4J_URI="***"
    NEO4J_USERNAME="***"
    NEO4J_PASSWORD="***"
    os.environ["NEO4J_URI"]=NEO4J_URI
    os.environ["NEO4J_USERNAME"]=NEO4J_USERNAME
    os.environ["NEO4J_PASSWORD"]=NEO4J_PASSWORD
    graph=Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)
    chain=GraphCypherQAChain.from_llm(llm=llm,graph=graph,verbose=True, allow_dangerous_requests=True, return_intermediate_steps=True)

    prompt_cypher="""
You are an expert in querying a Neo4j knowledge graph using Cypher. 
You will only return valid Cypher queries based on the provided schema.

Use concise MATCH patterns and RETURN only the relevant data. 
Use labels and relationships as defined in the schema. Do not guess any relationships or properties.

Schema:

Nodes:
- Room(room_number, room_type)
- ACUnit(label, purpose)
- TempSensor(label)
- OccSensor(label)

Relationships:
- (Room)-[:SERVICED_BY]->(ACUnit)
- (Room)-[:HAS_SENSOR]->(TempSensor)
- (TempSensor)-[:MONITORS]->(ACUnit)

Examples:
Q: Which rooms are serviced by AC-D2?
A: MATCH (r:Room)-[:SERVICED_BY]->(a:ACUnit {label: "AC-D2"}) RETURN r.room_number

Q: What AC units are monitored by temperature sensors in Room 101?
A: MATCH (r:Room {room_number: "101"})-[:HAS_SENSOR]->(t:TempSensor)-[:MONITORS]->(a:ACUnit) RETURN a.label
"""

    result = chain.invoke({
    "query": prompt_cypher + f"\nQ: {question}\nA:"
})
    context = result["intermediate_steps"][1]["context"]
    print("context:")
    print(context)

    if not context:
        return "No relevant data found in the knowledge graph."

    # Let the LLM summarize the answer
    summary_prompt = f"""You are an expert assistant.
    The user asked: "{question}"
    Here is the structured data returned from the Neo4j knowledge graph:
    {context}

    Please summarize the data in a natural language response.
    """

    try:
        return llm.invoke(summary_prompt)
    except Exception as e:
        return f"[ERROR] Failed to generate summary: {e}"

query_knowledge_graph_tool = Tool(
    name="query_knowledge_graph",
    func=query_knowledge_graph,
    # description="Querying the knowledge graph using a given question by the user. Knowledge graph contains information about the architeecture of the building. Neo4j: ACUnit(label), Room(room_number), relationship SERVICED_BY"
    description="Querying the knowledge graph using a given question by the user. Knowledge graph contains information about the architeecture of the building. Nodes: ACUnit, Room, TempSensor, OccSensor. Relationships: Room -HAS_SENSOR-> TempSensor, Room -HAS_SENSOR-> OccSensor, TempSensor -MONITORS-> ACUnit, Room -SERVICED_BY-> ACUnit"
)

query_sql_database_tool = Tool(
    name="query_sql_database",
    func=query_sql_database,
    description="Querying SQL database to using a question given by the user. The database contains time-series based occupancy and temperature sensor readings for each dorm rooms, in 5 mintute intervals. Tables: occupancy_sensors_week(Columns: timestamp, room, temperature), temperature_sensors_week(Columns: timestamp, room, occupancy)"
)

groq_api_key="***"
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.3-70b-versatile")

agent = initialize_agent(
    tools=[query_knowledge_graph_tool, query_sql_database_tool],
    llm=llm,
    agent="chat-conversational-react-description",
    memory=st.session_state.memory,
    verbose=True
)

#Streamlit Code
def strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_stdout" not in st.session_state:
    st.session_state.last_stdout = ""

st.title("LangChain Agent Chat")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

user_input = st.chat_input("Ask something...")

def capture_stdout(func, *args, **kwargs):
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    try:
        output = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return output, mystdout.getvalue()

if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    with st.spinner("Thinking..."):
        try:
            response, stdout = capture_stdout(agent.run, user_input)
        except Exception as e:
            response = f"[ERROR] {type(e).__name__}: {e}"
            stdout = ""
    st.session_state.chat_history.append({"role": "assistant", "text": response})
    clean_stdout = strip_ansi(stdout)
    st.session_state.last_stdout = clean_stdout
    st.rerun()

if st.session_state.last_stdout:
    with st.expander("Agent Logs / Tool Output"):
        st.code(st.session_state.last_stdout, language="bash")
