
# 🧠 AI Chatbot for Knowledge Graph & Time-Series Queries

A natural language chatbot that intelligently queries a **Neo4j knowledge graph** and a **TimescaleDB time-series database** using **LangChain agents**, **Groq LLMs**, and a **Streamlit interface**.

> Ask:  
> “Which rooms had temperature above 23°C on May 4?”  
> “Which AC unit monitors Room 106?”  
> “When was Room 101 last occupied?”

---

## ✨ Features

- 🔍 **Knowledge Graph (Neo4j)** querying via Cypher.
- 📈 **Time-Series (TimescaleDB)** querying via SQL.
- 🧠 **Multi-hop reasoning** across both databases using LangChain agent tools.
- 💬 **Conversational Memory** for context-aware interactions.
- 🖥️ **Streamlit Frontend** with GPT-style chat history and tool logs.
- ⚙️ Custom grounding prompts and summarization logic for accurate answers.

---

## 🏗️ Tech Stack

| Component         | Stack                                   |
|------------------|------------------------------------------|
| LLM Agent        | LangChain + Groq (LLaMA3 / Mixtral)      |
| Knowledge Graph  | Neo4j                                    |
| Time-Series DB   | TimescaleDB (PostgreSQL)                 |
| Backend Tools    | LangChain Tools & Chains (Cypher, SQL)   |
| Frontend         | Streamlit                                |
| LLM Enhancements | PromptTemplate, CoT Reasoning            |

---

## 🚀 Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/ai-kg-timeseries-chatbot.git
cd ai-kg-timeseries-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your environment variables

```bash
export GROQ_API_KEY=your_groq_key
export NEO4J_URI=neo4j+s://<your-uri>
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your_password
export TIMESCALEDB_URI=postgresql+psycopg2://user:pass@host:port/dbname
```

You can also store these in a `.env` file and load using `python-dotenv`.

### 4. Run the app

```bash
streamlit run agent.py
```

---

## 💡 Example Questions

```plaintext
• What rooms are serviced by AC-D2?
• Which rooms had temperature over 24°C on May 3rd?
• What is the average occupancy in Room 101 over the last 3 days?
• Which AC unit is monitored by Room 104’s temperature sensor?
```

---

## Future Work

- Add visual graph rendering for KG queries.
- Extend to support document retrieval (RAG).
- Integrate with Ollama or OpenRouter LLMs for local inference.

---
