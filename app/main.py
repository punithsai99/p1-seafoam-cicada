from langchain_core.messages import AIMessage
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import json, os, re

app = FastAPI(title="Phase 1 Mock API")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOCK_DIR = os.path.join(ROOT, "mock_data")

def load(name):
    with open(os.path.join(MOCK_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)

ORDERS = load("orders.json")
ISSUES = load("issues.json")
REPLIES = load("replies.json")

class TriageInput(BaseModel):
    ticket_text: str
    order_id: Optional[str] = None
class TriageState(TypedDict, total=False):
    messages: List[Dict[str, str]]
    ticket_text: str
    order_id: Optional[str]
    issue_type: str
    evidence: str
    recommendation: str
    order: Dict[str, Any]
    reply_text: str

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/orders/get")
def orders_get(order_id: str = Query(...)):
    for o in ORDERS:
        if o["order_id"] == order_id: return o
    raise HTTPException(status_code=404, detail="Order not found")

@app.get("/orders/search")
def orders_search(customer_email: Optional[str] = None, q: Optional[str] = None):
    matches = []
    for o in ORDERS:
        if customer_email and o["email"].lower() == customer_email.lower():
            matches.append(o)
        elif q and (o["order_id"].lower() in q.lower() or o["customer_name"].lower() in q.lower()):
            matches.append(o)
    return {"results": matches}

@app.post("/classify/issue")
def classify_issue(payload: dict):
    text = payload.get("ticket_text", "").lower()
    for rule in ISSUES:
        if rule["keyword"] in text:
            return {"issue_type": rule["issue_type"], "confidence": 0.85}
    return {"issue_type": "unknown", "confidence": 0.1}

def render_reply(issue_type: str, order: dict):
    template = next((r["template"] for r in REPLIES if r["issue_type"] == issue_type), None)
    if not template: template = "Hi {{customer_name}}, we are reviewing order {{order_id}}."
    return template.replace("{{customer_name}}", order.get("customer_name","Customer")).replace("{{order_id}}", order.get("order_id",""))

@app.post("/reply/draft")
def reply_draft(payload: dict):
    return {"reply_text": render_reply(payload.get("issue_type"), payload.get("order", {}))}

def ingest_node(state: TriageState) -> TriageState:
    text = state.get("ticket_text", "")
    order_id = state.get("order_id")

    messages = state.get("messages") or []
    if not messages:
        messages = [{"role": "customer", "content": text}]

    if not order_id:
        m = re.search(r"(ORD\d{4})", text, re.IGNORECASE)
        if m:
            order_id = m.group(1).upper()

    return {**state, "messages": messages, "ticket_text": text, "order_id": order_id}


def classify_issue_node(state: TriageState) -> TriageState:
    text = (state.get("ticket_text") or "").lower()

    issue_type = "unknown"
    evidence = "No keyword matched."
    recommendation = "Ask for more details or escalate."

    for rule in ISSUES:
        if rule["keyword"] in text:
            issue_type = rule["issue_type"]
            evidence = f"Matched keyword: {rule['keyword']}"
            recommendation = f"Handle as {issue_type}."
            break

    messages = state.get("messages") or []
    messages.append(
        {"role": "assistant", "content": f"I think this is '{issue_type}'. Evidence: {evidence}. Recommendation: {recommendation}"}
    )
    messages.append({"role": "admin", "content": "approved"})

    return {**state, "messages": messages, "issue_type": issue_type, "evidence": evidence, "recommendation": recommendation}


@tool
def fetch_order_tool(order_id: str) -> Dict[str, Any]:
    """Fetch a fake order by order_id from mock_data/orders.json."""
    order = next((o for o in ORDERS if o["order_id"] == order_id), None)
    if not order:
        raise ValueError("order not found")
    return order

def make_fetch_order_tool_call_node(state: TriageState) -> TriageState:
    order_id = state.get("order_id")
    if not order_id:
        return state

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "fetch_order_tool", "args": {"order_id": order_id}, "id": "call_fetch_order_1"}],
    )

    return {**state, "messages": (state.get("messages") or []) + [tool_call_msg]}


def draft_reply_node(state: TriageState) -> TriageState:
    order = state.get("order") or {}
    issue_type = state.get("issue_type") or "unknown"
    reply_text = render_reply(issue_type, order)

    messages = state.get("messages") or []
    messages.append({"role": "assistant", "content": reply_text})

    return {**state, "reply_text": reply_text, "messages": messages}

def store_order_from_tool_result_node(state: TriageState) -> TriageState:
    messages = state.get("messages") or []
    last = messages[-1] if messages else None

    order = None
    if last and hasattr(last, "content"):
        order = last.content

    if isinstance(order, str):
        try:
            order = json.loads(order)
        except Exception:
            pass

    if isinstance(order, dict):
        return {**state, "order": order}

    return state


tool_node = ToolNode([fetch_order_tool])

graph = StateGraph(TriageState)
graph.add_node("ingest", ingest_node)
graph.add_node("classify_issue", classify_issue_node)
graph.add_node("make_fetch_order_tool_call", make_fetch_order_tool_call_node)
graph.add_node("fetch_order", tool_node)
graph.add_node("store_order", store_order_from_tool_result_node)
graph.add_node("draft_reply", draft_reply_node)

graph.set_entry_point("ingest")
graph.add_edge("ingest", "classify_issue")


def route_after_classify(state: TriageState) -> str:
    if not state.get("order_id"):
        return END
    return "make_fetch_order_tool_call"


graph.add_conditional_edges(
    "classify_issue",
    route_after_classify,
    {"make_fetch_order_tool_call": "make_fetch_order_tool_call", END: END},
)

graph.add_edge("make_fetch_order_tool_call", "fetch_order")
graph.add_edge("fetch_order", "store_order")
graph.add_edge("store_order", "draft_reply")
graph.add_edge("draft_reply", END)

triage_graph = graph.compile()



@app.post("/triage/invoke")
def triage_invoke(body: TriageInput):
    # If order_id not provided, try extracting from text
    order_id = body.order_id
    if not order_id:
        m = re.search(r"(ORD\d{4})", body.ticket_text or "", re.IGNORECASE)
        if m:
            order_id = m.group(1).upper()

    # ✅ REQUIRED by tests: if still missing, return 400
    if not order_id:
        raise HTTPException(status_code=400, detail="order_id missing and not found in text")

    init_state: TriageState = {
        "ticket_text": body.ticket_text,
        "order_id": order_id,
        "messages": [{"role": "customer", "content": body.ticket_text}],
    }

    result = triage_graph.invoke(init_state)

    if "order" not in result:
        raise HTTPException(status_code=404, detail="order not found")

    return {
        "order_id": result.get("order_id"),
        "issue_type": result.get("issue_type"),
        "evidence": result.get("evidence"),
        "recommendation": result.get("recommendation"),
        "order": result.get("order"),
        "reply_text": result.get("reply_text"),
        "messages": result.get("messages"),
    }
