# MongoDB + RAG + LangChain Memory Refactor

This project is a **Flask-based AI application** that integrates:
- **MongoDB Atlas** for data storage and vector search  
- **LangChain + LangGraph** for conversational reasoning, tool orchestration, and memory  
- **OpenAI GPT-4o-mini** for intelligent response generation  
- **OpenRouteService (ORS)** for geocoding and route analysis  
- **Microsoft Graph API** for Outlook email integration  

It provides Retrieval-Augmented Generation (RAG), vector search, contextual insights, and AI-assisted outreach — all backed by persistent chat memory in MongoDB.

---

## 🚀 Features

- **MongoDB Vector Search + Fallback**  
  Uses Atlas Vector Search when available; otherwise falls back to local cosine similarity.  

- **Retrieval-Augmented Generation (RAG)**  
  Optimized for concise, token-efficient client data queries.  

- **LangGraph Orchestration**  
  - Supervisor node controls reasoning  
  - Custom tool node executes external API/tools  
  - Conditional edges for looping execution  

- **AI Tools Integrated**
  - `get_clients_on_route`: Finds clients near a driving route using ORS + MongoDB.  
  - `query_clients_rag`: RAG search over MongoDB client data.  
  - `get_contextual_insights`: Fetches client-specific info & drafts outreach emails.  
  - `get_outlook_interactions`: Summarizes Outlook email threads via MS Graph API.  

- **Persistent Memory**  
  Uses `MongoDBChatMessageHistory` + `ConversationSummaryBufferMemory` to maintain contextual dialogue while minimizing token cost.  

- **Secure Auth & APIs**
  - OAuth2 login via Microsoft Identity Platform (MSAL).  
  - Environment-driven configuration.  

---

## 📂 Project Structure

.
├── app.py # Main Flask application
├── templates/
│ └── index.html # Simple frontend entrypoint
├── requirements.txt # Python dependencies
├── .env.example # Example environment variables
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-repo/mongo-rag-langgraph.git
   cd mongo-rag-langgraph
Create & activate virtual environment

bash
Copy code
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies

bash
Copy code
pip install -r requirements.txt
🔑 Environment Variables
Create a .env file with the following keys (see .env.example):

ini
Copy code
# OpenAI
OPENAI_API_KEY=your_openai_key

# OpenRouteService
ORS_API_KEY=your_ors_key

# MongoDB
MONGO_URI=mongodb+srv://...
DB_NAME=client_outreach
MONGO_COLLECTION=municipalities
VECTORS_COLLECTION=vectors

# Microsoft Graph (OAuth)
CLIENT_ID=your_ms_client_id
CLIENT_SECRET=your_ms_client_secret
TENANT_ID=your_tenant_id
▶️ Running the App
bash
Copy code
flask run
Then open in browser:

cpp
Copy code
http://127.0.0.1:5000/
🧠 Core Workflow
User interacts with the Flask frontend.

Conversation flows through LangGraph Agent.

Supervisor decides whether to call tools or respond directly.

Tools fetch data from MongoDB, ORS, or MS Graph.

Responses are summarized and stored in MongoDB-backed memory.

📌 Example Use Cases
Find all municipalities near a driving route between two cities.

Query client database (CFOs, Mayors, Admins) using natural language.

Draft outreach emails tailored to each municipality contact.

Summarize recent email interactions with a client from Outlook.

🛠️ Tech Stack
Backend: Flask, Flask-Session, Flask-CORS

Database: MongoDB (Atlas Vector Search + Document Store)

LLMs & Tools: LangChain, LangGraph, OpenAI GPT

Routing API: OpenRouteService

Email API: Microsoft Graph (via MSAL)

📜 Logging
The app uses Python logging with INFO level. Logs include:

MongoDB queries & results

ORS API usage

Tool execution details

Vector search fallbacks

🔮 Next Steps
Add Dockerfile for containerization

Expand frontend with chat UI

Enhance error handling for external APIs

Implement multi-user session management

📄 License
MIT License. Free to use and modify.
