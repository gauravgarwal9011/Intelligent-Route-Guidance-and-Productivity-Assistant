# app.py
# MongoDB + RAG + LangChain memory refactor

import os
import json
import time
import uuid
import math
import logging
import datetime
from typing import TypedDict, Annotated, Sequence, List, Dict, Any

from dotenv import load_dotenv
from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from flask_cors import CORS
from flask_session import Session
import requests
import msal
import pandas as pd

import openrouteservice
from openrouteservice.exceptions import ApiError

# LangChain / LangGraph
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END, START
from langchain_core.runnables import RunnableLambda

# Memory (Mongo-backed)
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.docstore.document import Document

# Vector search (MongoDB Atlas Vector Search if available)
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# --- Config / Env ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ORS_API_KEY = os.getenv("ORS_API_KEY")

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TENANT_ID = os.getenv("TENANT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
GRAPH_ENDPOINT = "https://graph.microsoft.com/v1.0"

SCOPES = ["User.Read", "Mail.Read"]
REDIRECT_PATH = "/get_token"

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "client_outreach")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "municipalities")
# Optional: separate vector collection; if not provided, we'll use the same collection with a `embedding` field
VECTORS_COLLECTION = os.getenv("VECTORS_COLLECTION", "vectors")

# --- Flask App ---
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.urandom(24)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# --- Services ---
if not MONGO_URI:
    logger.warning("MONGODB_URI is not set. The app will not be able to load clients.")
mongo_client = MongoClient(MONGO_URI) if MONGO_URI else None
mongo_db = mongo_client[DB_NAME] if DB_NAME else None
clients_col = mongo_db[MONGO_COLLECTION] if MONGO_COLLECTION else None
vectors_col = mongo_db[VECTORS_COLLECTION] if VECTORS_COLLECTION else None

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# --- Utilities ---
def json_serial(obj):
    if isinstance(obj, (datetime.datetime, datetime.date, pd.Timestamp)):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    raise TypeError(f"Type {type(obj)} not serializable")

def haversine_km(lat1, lon1, lat2, lon2):
    # distance between two lat/lon points
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def min_distance_point_to_polyline_km(point_lat, point_lon, polyline_latlon: List[List[float]]):
    # Approximate by sampling: min haversine to every vertex (fast and sufficient for our case)
    return min(haversine_km(point_lat, point_lon, lat, lon) for lat, lon in polyline_latlon)

def mongo_get_clients() -> List[Dict[str, Any]]:
    """
    Get clients from MongoDB with improved error handling and logging.
    """
    logger.info("Attempting to fetch clients from MongoDB...")
    
    if not MONGO_URI:
        logger.error("MONGO_URI environment variable is not set")
        return []
    
    if mongo_client is None:
        logger.error("MongoDB client is not initialized")
        return []
        
    if mongo_db is None:
        logger.error("MongoDB database is not initialized")
        return []
        
    if clients_col is None:
        logger.error("MongoDB collection is not initialized")
        return []

    try:
        # Test connection first
        mongo_client.admin.command('ismaster')
        logger.info("MongoDB connection is alive")
        
        # Check if collection exists
        collections = mongo_db.list_collection_names()
        if MONGO_COLLECTION not in collections:
            logger.error(f"Collection '{MONGO_COLLECTION}' does not exist. Available collections: {collections}")
            return []
        
        # Count documents first
        doc_count = clients_col.count_documents({})
        logger.info(f"Total documents in collection: {doc_count}")
        
        if doc_count == 0:
            logger.warning("Collection is empty")
            return []
        
        # Expecting documents with Latitude, Longitude and other fields
        # Minimal projection to reduce payload
        projection = {
            "Municipality": 1, "County": 1, "State": 1, "Country": 1,
            "CFO": 1, "CFO Email": 1, "Mayor": 1, "Mayor Email": 1,
            "Administrator": 1, "Administrator Email": 1,
            "latitude": 1, "longitude": 1
        }
        
        # Execute the query
        cursor = clients_col.find({}, projection)
        results = list(cursor)
        
        logger.info(f"Query returned {len(results)} documents")
        
        # Log some stats about the data
        if results:
            coords_count = sum(1 for doc in results if doc.get('latitude') and doc.get('longitude'))
            logger.info(f"Documents with coordinates: {coords_count}/{len(results)}")
            
            # Sample first document structure
            sample_doc = results[0]
            logger.info(f"Sample document keys: {list(sample_doc.keys())}")
        
        return results
        
    except PyMongoError as e:
        logger.exception(f"MongoDB error in mongo_get_clients: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error in mongo_get_clients: {e}")
        return []

def record_to_doc(r: Dict[str, Any]) -> Document:
    # Build a concise text for RAG that costs fewer tokens
    lines = []
    add = lines.append
    add(f"Municipality: {r.get('Municipality','N/A')}")
    add(f"County: {r.get('County','N/A')}, State: {r.get('State','N/A')}, Country: {r.get('Country','N/A')}")
    add(f"CFO: {r.get('CFO','N/A')} ({r.get('CFO Email','N/A')})")
    add(f"Mayor: {r.get('Mayor','N/A')} ({r.get('Mayor Email','N/A')})")
    add(f"Administrator: {r.get('Administrator','N/A')} ({r.get('Administrator Email','N/A')})")
    txt = "\n".join(lines)
    meta = {
        "Municipality": r.get("Municipality"),
        "County": r.get("County"), "State": r.get("State"), "Country": r.get("Country"),
        "CFO": r.get("CFO"), "CFO Email": r.get("CFO Email"),
        "Mayor": r.get("Mayor"), "Mayor Email": r.get("Mayor Email"),
        "Administrator": r.get("Administrator"), "Administrator Email": r.get("Administrator Email"),
        "Latitude": r.get("Latitude"), "Longitude": r.get("Longitude"),
        "_id": str(r.get("_id"))
    }
    return Document(page_content=txt, metadata=meta)

def ensure_vector_for_record(r: Dict[str, Any]):
    """Upsert an embedding for this record into vectors_col (Atlas Vector Search) or add to same collection."""
    if vectors_col is None:
        return
    if "Municipality" not in r:
        return
    try:
        doc_id = str(r["_id"])
        text = record_to_doc(r).page_content
        emb = embeddings.embed_query(text)
        vectors_col.update_one(
            {"_id": doc_id},
            {"$set": {
                "text": text,
                "embedding": emb,
                "metadata": {k: v for k, v in r.items() if k != "embedding"}
            }},
            upsert=True
        )
    except Exception as e:
        logger.warning(f"Failed to upsert vector for {_safe(r, '_id')}: {e}")

def _safe(d: Dict[str, Any], k: str):
    try:
        return d[k]
    except Exception:
        return None

def mongo_vector_search(query: str, k: int = 6) -> List[Document]:
    """
    Try MongoDB Atlas Vector Search; if unavailable, fall back to local embedding + brute force cosine sim
    over a small set (still OK for a few thousand docs).
    """
    if clients_col is None:   # ✅ safe check
        logger.error("No Mongo collection available.")
        return []

    # If we have a vector collection and it contains embeddings, use $vectorSearch
    try:
        q_emb = embeddings.embed_query(query)
        # Fix: Use 'is not None' instead of truthy check on collection
        if vectors_col is not None:
            # Attempt vector search (Atlas Vector Search 2.0)
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",          # Ensure you created an index named 'vector_index'
                        "path": "embedding",              # Changed from "emb" to "embedding"
                        "queryVector": q_emb,
                        "numCandidates": 100,
                        "limit": k
                    }
                },
                {"$project": {"text": 1, "metadata": 1, "_id": 1}}
            ]
            res = list(vectors_col.aggregate(pipeline))
            docs = []
            for r in res:
                md = r.get("metadata", {})
                md["_id"] = str(r.get("_id"))
                docs.append(Document(page_content=r.get("text",""), metadata=md))
            if docs:
                return docs
    except Exception as e:
        logger.info(f"Vector search not available or failed; fallback to ephemeral search. Reason: {e}")

    # Fallback: embed on the fly and compute similarity locally
    try:
        all_records = mongo_get_clients()
        if not all_records:
            return []
        # Build candidates (limit for efficiency)
        candidates = all_records[:3000]  # adjustable cap
        texts = [record_to_doc(r) for r in candidates]
        qv = embeddings.embed_query(query)
        vecs = embeddings.embed_documents([d.page_content for d in texts])

        # cosine similarity
        def cos(a, b):
            dot = sum(x*y for x,y in zip(a,b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(y*y for y in b))
            return dot / (na*nb + 1e-8)

        scored = [(cos(qv, v), d) for v, d in zip(vecs, texts)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]
    except Exception as e:
        logger.exception(f"Ephemeral vector search failed: {e}")
        return []

def mongo_vector_retriever():
    def _search(query: str):
        return mongo_vector_search(query)
    return RunnableLambda(_search)

def compressing_retriever():
    base = mongo_vector_retriever()
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base,
    )

# --- Tools ----------------------------------------------------------------------------------
@tool
def get_clients_on_route(start_location: str, end_location: str, max_distance_km: int = 30) -> str:
    """Find clients in MongoDB within ~max_distance_km of the ORS driving route between two textual places."""
    logger.info(f"Tool 'get_clients_on_route' start='{start_location}', end='{end_location}', max_km={max_distance_km}")
    
    try:
        if not ORS_API_KEY:
            return "Error: OpenRouteService API key not configured."

        ors_client = openrouteservice.Client(key=ORS_API_KEY)
        
        # Geocode the two endpoints
        def geocode_once(text: str):
            res = ors_client.pelias_search(text=text, size=1)
            if res and res.get("features"):
                c = res["features"][0]["geometry"]["coordinates"]  # [lon, lat]
                return c
            return None

        start_coords = geocode_once(start_location)
        if not start_coords:
            return f"Error: Could not geocode '{start_location}'."
            
        end_coords = geocode_once(end_location)
        if not end_coords:
            return f"Error: Could not geocode '{end_location}'."

        logger.info(f"Geocoded: {start_location} -> {start_coords}, {end_location} -> {end_coords}")

        # Get route
        route_response = ors_client.directions(
            coordinates=[start_coords, end_coords],
            profile='driving-car',
            format='geojson', 
            geometry_simplify=False, 
            geometry=True
        )
        
        route_coords = route_response['features'][0]['geometry']['coordinates']  # [[lon,lat], ...]
        route_latlon = [[c[1], c[0]] for c in route_coords]  # Convert to [lat, lon]
        
        logger.info(f"Route calculated with {len(route_coords)} points")

        # Load clients from MongoDB - WITH DETAILED LOGGING
        logger.info("Loading clients from MongoDB...")
        records = mongo_get_clients()
        
        if not records:
            logger.warning("No records returned from mongo_get_clients()")
            return json.dumps({
                "route": route_coords, 
                "clients_df_json": "[]", 
                "llm_summary": {
                    "clients_table": "No clients in database.", 
                    "message": "No clients found in the database."
                }
            })

        logger.info(f"Loaded {len(records)} client records")

        # Find clients near the route
        nearby = []
        records_with_coords = 0
        
        for i, r in enumerate(records):
            lat, lon = r.get("latitude"), r.get("longitude")
            
            if lat is None or lon is None:
                if i < 5:  # Log first 5 missing coordinates
                    logger.warning(f"Record {i} missing coordinates: Municipality={r.get('Municipality')}, lat={lat}, lon={lon}")
                continue
                
            records_with_coords += 1
            
            try:
                dmin = min_distance_point_to_polyline_km(lat, lon, route_latlon)
                
                if dmin <= max_distance_km:
                    item = {
                        "CFO": r.get("CFO"),
                        "Mayor": r.get("Mayor"),
                        "Administrator": r.get("Administrator"),
                        "Municipality": r.get("Municipality"),
                        "County": r.get("County"),
                        "State": r.get("State"),
                        "Country": r.get("Country"),
                        "coords": [lon, lat],
                        "distance_km": round(dmin, 2)
                    }
                    nearby.append(item)
            except Exception as e:
                logger.warning(f"Error calculating distance for record {i}: {e}")

        logger.info(f"Records with coordinates: {records_with_coords}/{len(records)}")
        logger.info(f"Clients found within {max_distance_km}km: {len(nearby)}")

        # Sort by distance
        nearby.sort(key=lambda x: x["distance_km"])
        
        if not nearby:
            message = f"No clients found within {max_distance_km} km of the route from {start_location} to {end_location}."
            if records_with_coords == 0:
                message += " Note: No client records have coordinate data."
            
            out = {
                "route": route_coords, 
                "clients_df_json": "[]", 
                "llm_summary": {
                    "clients_table": "No clients were found along the specified route.", 
                    "message": message
                }
            }
            return json.dumps(out, default=json_serial)

        # Create summary table
        df = pd.DataFrame(nearby)[["CFO","Mayor","Administrator","Municipality", "County", "State", "distance_km"]]
        from tabulate import tabulate
        clients_table = tabulate(df, headers="keys", tablefmt="psql", showindex=False)
        
        llm_summary = {
            "clients_table": clients_table,
            "message": f"I found {len(nearby)} clients along the route from {start_location} to {end_location}."
        }
        
        result = {
            "route": route_coords, 
            "clients_df_json": json.dumps(nearby, default=json_serial), 
            "llm_summary": llm_summary
        }
        
        return json.dumps(result, default=json_serial)

    except ApiError as api_err:
        logger.error(f"OpenRouteService API error: {api_err}")
        return f"Error: OpenRouteService API error - {str(api_err)}"
    except Exception as e:
        logger.exception("Unexpected error in get_clients_on_route")
        return f"An unexpected error occurred while finding clients along the route: {str(e)}"

@tool
def query_clients_rag(question: str) -> str:
    """
    Optimized token-efficient RAG over MongoDB client data.
    Returns a concise summary + top matches with fewer API calls.
    """
    try:
        # Use simple vector search without compression (saves 6+ API calls)
        docs = mongo_vector_search(question, k=8)  # Get more initially
        if not docs:
            return "No relevant results."

        # Build a very compact answer - no LLM compression step
        seen = set()
        hits = []
        for d in docs:
            muni = d.metadata.get("Municipality") or "N/A"
            if muni in seen:
                continue
            seen.add(muni)
            
            # Filter specifically for the query - simple keyword matching
            doc_text = d.page_content.lower()
            query_lower = question.lower()
            
            # Simple relevance scoring based on keyword matches
            keywords = ['cfo', 'katie', 'administrator', 'mayor', 'email']
            matches = sum(1 for kw in keywords if kw in query_lower and kw in doc_text)
            
            hit = {
                "Municipality": muni,
                "County": d.metadata.get("County"),
                "State": d.metadata.get("State"),
                "Country": d.metadata.get("Country"),
                "CFO": d.metadata.get("CFO"),
                "CFO Email": d.metadata.get("CFO Email"),
                "Mayor": d.metadata.get("Mayor"),
                "Mayor Email": d.metadata.get("Mayor Email"),
                "Administrator": d.metadata.get("Administrator"),
                "Administrator Email": d.metadata.get("Administrator Email"),
                "relevance_score": matches
            }
            hits.append(hit)
            
            if len(hits) >= 6:  # Limit results
                break

        # Sort by relevance and take top results
        hits.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        top_hits = hits[:5]

        # Generate summary with ONE API call instead of 6+
        summary_prompt = (
            f"Based on the following client matches, provide a brief answer to: '{question}'\n"
            f"Be concise and focus on the most relevant information.\n\n"
            f"Matches:\n{json.dumps(top_hits, indent=2)}"
        )
        answer = llm.invoke(summary_prompt).content
        
        return json.dumps({
            "summary": answer,
            "matches": top_hits,
            "total_found": len(hits)
        }, default=json_serial)
        
    except Exception as e:
        logger.exception("RAG query failed")
        return f"RAG query failed: {str(e)}"
    

@tool
def get_contextual_insights(client_name: str) -> str:
    """Look up a single client by Municipality and return compact insights + a draft email."""
    try:
        if clients_col is None:
            logger.error("No Mongo collection available.")
            return "No MongoDB collection available."
        
        # Search by Municipality (primary) or try other fields as fallback
        rec = None
        
        # Try Municipality first
        rec = clients_col.find_one({"Municipality": {"$regex": f"^{client_name}$", "$options": "i"}})
        
        # If not found, try partial match on Municipality
        if not rec:
            rec = clients_col.find_one({"Municipality": {"$regex": client_name, "$options": "i"}})
        
        # If still not found, try searching by CFO name
        if not rec:
            rec = clients_col.find_one({"CFO": {"$regex": client_name, "$options": "i"}})
        
        # If still not found, try searching by Mayor name
        if not rec:
            rec = clients_col.find_one({"Mayor": {"$regex": client_name, "$options": "i"}})
        
        if not rec:
            return f"Client '{client_name}' was not found in any field (Municipality, CFO, Mayor)."
        
        doc = record_to_doc(rec)
        
        # Try to ensure vector exists (but don't fail if it doesn't work)
        try:
            if vectors_col is not None:  # Fixed boolean check
                ensure_vector_for_record(rec)
        except Exception as ve:
            logger.warning(f"Failed to create vector for {client_name}: {ve}")

        # Determine the primary contact
        primary_contact = "CFO" if rec.get("CFO") else "Mayor" if rec.get("Mayor") else "Administrator"
        primary_email = rec.get("CFO Email") if rec.get("CFO") else rec.get("Mayor Email") if rec.get("Mayor") else rec.get("Administrator Email")
        
        draft_prompt = (
            f"Write a short, friendly outreach note (<=120 words) to the {primary_contact}. "
            "Keep it neutral, no hard sell, 3 short paragraphs max. "
            f"Municipality: {rec.get('Municipality', 'N/A')}\n"
            f"Contact: {rec.get(primary_contact, 'N/A')} ({primary_email or 'No email'})\n"
            f"Additional context:\n{doc.page_content}"
        )
        
        draft = llm.invoke(draft_prompt).content
        
        insights = {
            "Municipality": rec.get("Municipality", "N/A"),
            "County": rec.get("County", "N/A"),
            "State": rec.get("State", "N/A"),
            "CFO": rec.get("CFO", "N/A"),
            "CFO Email": rec.get("CFO Email", "N/A"),
            "Mayor": rec.get("Mayor", "N/A"),
            "Mayor Email": rec.get("Mayor Email", "N/A"),
            "Administrator": rec.get("Administrator", "N/A"),
            "Administrator Email": rec.get("Administrator Email", "N/A"),
            "Primary_Contact": primary_contact,
            "Primary_Email": primary_email or "N/A",
            "Draft_Email": draft
        }
        return json.dumps(insights, indent=2, default=json_serial)
        
    except Exception as e:
        logger.exception("contextual insights failed")
        return f"Failed to fetch insights: {str(e)}"

@tool
def get_outlook_interactions(search_query: str) -> str:
    """
    Search Outlook and return a concise, **summarized** digest.
    """
    logger.info(f"Tool 'get_outlook_interactions' q='{search_query}'")
    try:
        if "access_token" not in session:
            return "Error: You must be logged in to search Outlook emails."
        access_token = session["access_token"]
        headers = {'Authorization': 'Bearer ' + access_token}
        endpoint = f"{GRAPH_ENDPOINT}/me/messages?$search=\"{search_query}\"&$top=5"

        resp = requests.get(endpoint, headers=headers)
        resp.raise_for_status()
        emails = resp.json().get('value', [])

        if not emails:
            return f"No emails were found matching '{search_query}'."

        # Extract light fields to keep token cost low
        light = [{
            "subject": e.get("subject"),
            "from": (e.get("from", {}) or {}).get("emailAddress", {}).get("address"),
            "received": e.get("receivedDateTime"),
            "preview": e.get("bodyPreview")
        } for e in emails]

        prompt = (
            "You summarize emails **concisely**. Produce:\n"
            "1) 4–6 bullet highlights (group by topic if possible),\n"
            "2) 2 concrete next actions.\n"
            "Keep it under 120 words total.\n\n"
            f"Emails:\n{json.dumps(light, indent=2)}"
        )
        digest = llm.invoke(prompt).content
        return json.dumps({"emails": light, "summary": digest}, indent=2)
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error searching Outlook: {http_err} - {getattr(http_err.response, 'text', '')}")
        return "Error: Failed to communicate with the Outlook service."
    except Exception as e:
        logger.exception("Unexpected error in get_outlook_interactions")
        return "An unexpected error occurred while searching your emails."

# --- LangGraph + Memory ---------------------------------------------------------------------- 

def message_to_dict(message):
    if isinstance(message, HumanMessage):
        return {"type": "human", "content": message.content}
    elif isinstance(message, AIMessage):
        tool_calls = [{"name": tc.get('name'), "args": tc.get('args'), "id": tc.get('id')} for tc in (message.tool_calls or [])]
        return {"type": "ai", "content": message.content, "tool_calls": tool_calls}
    elif isinstance(message, ToolMessage):
        return {"type": "tool", "content": message.content, "tool_call_id": message.tool_call_id, "name": message.name}
    else:
        return {"type": "unknown", "content": str(message)}

def dict_to_message(d):
    msg_type = d.get("type")
    if msg_type == "human":
        return HumanMessage(content=d["content"])
    elif msg_type == "ai":
        tool_calls = [{"name": tc.get('name'), "args": tc.get('args'), "id": tc.get('id')} for tc in d.get("tool_calls", [])]
        return AIMessage(content=d["content"], tool_calls=tool_calls)
    elif msg_type == "tool":
        return ToolMessage(content=d["content"], tool_call_id=d["tool_call_id"], name=d.get("name"))
    else:
        return HumanMessage(content=d.get("content", ""))

# --- State definition (simplified) --------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

tools = [
    get_clients_on_route,
    query_clients_rag,
    get_contextual_insights,
    get_outlook_interactions
]

def custom_tool_node(state):
    logger.info("Executing tool node...")
    tool_messages = []
    last_message = state["messages"][-1]
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        logger.info(f"Running tool '{tool_name}' with args: {tool_args}")
        tool_found = False
        for tool_func in tools:
            if tool_func.name == tool_name:
                tool_found = True
                response_str = tool_func.invoke(tool_args)
                tool_messages.append(ToolMessage(content=response_str, tool_call_id=tool_call["id"], name=tool_name))
                break
        if not tool_found:
            tool_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call["id"], name=tool_name))
    return {"messages": tool_messages}

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY).bind_tools(tools)

def supervisor_node(state):
    logger.info("Supervisor node invoking the model.")
    return {"messages": [model.invoke(state["messages"])]}

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("tools", custom_tool_node)

def should_continue(state: AgentState):
    if state['messages'] and getattr(state['messages'][-1], "tool_calls", None):
        return "tools"
    return "end"

workflow.add_conditional_edges("supervisor", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "supervisor")
workflow.add_edge(START, "supervisor")
agent_app = workflow.compile()
logger.info("LangGraph agent compiled.")

# --- MongoDB Chat History -----------------------------------------------------------------------------------
def get_history(session_id: str) -> MongoDBChatMessageHistory:
    """Returns a MongoDB-backed chat message history."""
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_URI,
        database_name=DB_NAME,
        collection_name="chat_history"
    )

def get_memory(session_id: str):
    # Summary buffer to keep tokens low
    return ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=get_history(session_id),
        max_token_limit=1000,  # Keep small to reduce cost
        return_messages=True
    )


# --- Flask Routes ------------------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    logger.info("Login process initiated.")
    session["state"] = str(uuid.uuid4())
    auth_url = _build_auth_url(scopes=SCOPES, state=session["state"])
    return redirect(auth_url)

@app.route(REDIRECT_PATH)
def get_token():
    logger.info("Handling redirect from Microsoft login.")
    if request.args.get('state') != session.get('state'):
        return "State does not match. Authentication failed.", 400
    if request.args.get('code'):
        cache = _load_cache()
        msal_app = _build_msal_app(cache=cache)
        result = msal_app.acquire_token_by_authorization_code(
            request.args['code'], scopes=SCOPES, redirect_uri=url_for("get_token", _external=True)
        )
        if "access_token" in result:
            session["access_token"] = result["access_token"]
            _save_cache(cache)
            return """
            <script>window.opener.postMessage("login_successful", "*");window.close();</script>
            Login successful! You can close this window.
            """
    return "Failed to acquire token.", 400

@app.route('/invoke_agent', methods=['POST'])
def invoke_agent():
    try:
        data = request.get_json() or {}
        query = data.get('query')
        session_id = data.get('session_id')  # provided by front-end localStorage
        if not query:
            return jsonify({"error": "Query is required."}), 400
        if not session_id:
            return jsonify({"error": "Missing session_id."}), 400

        memory = get_memory(session_id)

        # Build conversation from memory + new user msg
        past = memory.load_memory_variables({}).get("history", [])
        conversation_messages: List[BaseMessage] = past + [HumanMessage(content=query)]
        num_messages_before = len(conversation_messages)

        final_state = agent_app.invoke({"messages": conversation_messages})

        # Persist the turn into memory
        # We store *all* messages during this step compactly
        # last message is AI
        for m in final_state["messages"][num_messages_before-1:]:
            if isinstance(m, HumanMessage):
                memory.chat_memory.add_user_message(m.content)
            elif isinstance(m, ToolMessage):
                # Store a condensed tool output (to keep history cheap)
                truncated = m.content
                if isinstance(truncated, str) and len(truncated) > 1200:
                    truncated = truncated[:1200] + " ...[truncated]"
                memory.chat_memory.add_ai_message(f"[{m.name}] {truncated}")
            elif isinstance(m, AIMessage):
                memory.chat_memory.add_ai_message(m.content)

        # Compute final response for frontend
        final_response_content = final_state['messages'][-1].content
        # If route tool was executed this turn, send its payload to frontend
        for message in final_state['messages'][num_messages_before:]:
            if isinstance(message, ToolMessage) and message.name == "get_clients_on_route":
                final_response_content = message.content
                break

        return jsonify({"response": final_response_content})
    except Exception as e:
        logger.exception("invoke_agent fatal error")
        return jsonify({"error": "An unexpected server error occurred. Please check the logs."}), 500

@app.route('/test_mongo')
def test_mongo():
    try:
        records = mongo_get_clients()
        return jsonify({
            "count": len(records),
            "sample": records[:2] if records else [],
            "mongo_uri_set": bool(MONGO_URI),
            "db_name": "client_outreach",
            "collection_name": "municipalities",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# --- MSAL Helpers ------------------------------------------------------------------------------------
def _build_msal_app(cache=None):
    return msal.ConfidentialClientApplication(
        CLIENT_ID, authority=AUTHORITY,
        client_credential=CLIENT_SECRET, token_cache=cache
    )

def _build_auth_url(scopes=None, state=None):
    msal_app = _build_msal_app()
    return msal_app.get_authorization_request_url(
        scopes or [], state=state or str(uuid.uuid4()),
        redirect_uri=url_for("get_token", _external=True)
    )

def _load_cache():
    cache = msal.SerializableTokenCache()
    if session.get("token_cache"):
        cache.deserialize(session["token_cache"])
    return cache

def _save_cache(cache):
    if cache.has_state_changed:
        session["token_cache"] = cache.serialize()

def _get_token_for_scopes(scopes):
    cache = _load_cache()
    msal_app = _build_msal_app(cache=cache)
    accounts = msal_app.get_accounts()
    if not accounts:
        return {"error": "login_required", "error_description": "No user is logged in."}
    result = msal_app.acquire_token_silent(scopes, account=accounts[0])
    if not result:
        return {"error": "relogin_required", "error_description": "Could not acquire token silently."}
    _save_cache(cache)
    return result

if __name__ == '__main__':
    logger.info("Starting Flask dev server.")
    app.run(debug=True, port=5001)
