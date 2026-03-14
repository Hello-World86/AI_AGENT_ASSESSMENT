import os
import json
import re 
from typing import Any, List, Dict, Optional

from tools import get_order_details, get_customer_profile
from rag import retrieve_policy
from model import get_model

TOOL_REGISTRY = {
    "get_order_details": get_order_details,
    "get_customer_profile": get_customer_profile,
    "retrieve_policy": retrieve_policy
}

TOOL_DEFINITIONS = [
    {
    "type": "function",
    "function": {
        "name": "get_order_details",
        "description":(
            "Look up a specific order in the database by order ID."
            "Use this when the user references an order number like ('Order #2022','order 2022', or '#2022')"
            "Return order status, customer type (VIP or Standard), and number of dats since delivery."
        ),
        "parameters":{
            "type":"object",
            "properties":{
                "order_id":{
                    "type":"string",
                    "description": "The order ID mentioned by the user (e.g., '8909' )."
                }
            },
            "required": ["order_id"]
        }
    }
},
    {
        "type": "function",
        "function": {
            "name": "get_customer_profile",
            "description": (
                "Fetch the customer profile associated with a specific order, "
                "including whether the customer is VIP or Standard. "
                "Use this tool when you need information about customer tier benefits, "
                "support privileges, or additional perks beyond the general policy rules. "
                "Input must be either 'VIP' or 'Standard'."
        ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_type": {
                        "type": "string",
                        "description": "The customer tier: 'VIP' or 'Standard'",
                    }
                },
                "required": ["customer_type"],
            },
        },
    },

 {
     "type": "function",
     "function":{
         "name":"retrieve_policy",
         "description":(
             "Search the customer service knowledge base for company policies."
             "Use this for questions about return windows, refund policies,"
             "delay compensation, or damage claims."
         ),
         "parameters":{
             "type":"object",
             "properties":{
                 "query":{
                     "type":"string",
                     "description":"The policy topic to search for (example: 'VIP return policy')."
                 }
             },
             "required":["query"]
            }
        }
    }
]
# Keywords to detect policy-related queries
POLICY_KEYWORDS = [
    "policy", "return", "refund", "compensation", "delay", "damage",
    "eligibility", "return window", "refund policy", "delay compensation",
    "damage claim", "eligible for return", "eligible for refund","eligible for compensation",
    "return policy", "refund eligibility", "compensation eligibility",
    "based on the policy", "according to the policy", "what is the policy for", "what's the policy for",
    "can I return", "can I get a refund", "can I get compensation",
    "how long is the return window", "what is the return window"
]

#system prompt for the agent(master prompt)
PROMPT = """
You are a concise conversational AI agent that answers customers queries.
Your job is to answer customer questions about orders, delivery status and company policies.
You have access to tools that allow you to retrieve structured data from a CRM tool and retrieve company policies from a RAG.
Follow these rules strictly. Provide answer in plain facts and no elaborations.

You have access to the following tools:

Tool access:
1. get_order_details
2. get_customer_profile
3. retrieve_policy

Always follow this reasoning approach:
Step 1: Understand the user's question.
Step 2: Determine whether you need to retrieve:
- structured data (use get_order_details/get_customer_profile),
- policy information(use retrieve_policy)
- or both.
Step 3: If an order number is mentioned, call get_order_details first.
Step 4: If the question mentions customer type or you need to determine eligibility for policies, call get_customer_profile with the customer type from the order details.
Step 5: If the question involves return eligibility or refunds, retrieve the relevant policy using retrieve_policy.
Step 6: Combine the information from the order data and policy to produce a clear answer.

IMPORTANT RULES
Never guess order information.
Always use tools when order data is required. 
Do not guess or invent policies.

If the user asks about the return eligibility, you must check the policy rules only.
"""

def _needs_policy(query:str)->bool:
    """"Determine if the user's query is related to company policies based on keyword matching."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in POLICY_KEYWORDS)

def _run_tool(tool_name:str, parameters:dict)-> dict:
    """Execute a tool dynamically."""
    if tool_name in TOOL_REGISTRY:
        return TOOL_REGISTRY[tool_name](**parameters)
    return {"error": f"Unknown tool: {tool_name}"}

def run_agent(
        query: str,
        max_iterations: int = 10,
        chat_history: Optional[List[Dict[str, str]]] = None
) -> dict:
   
    """Main agent function for processing user queries and generating responses using tools."""
    llm = get_model()
    steps: list[dict] = []

    has_order_number = re.search(r"#\s*(\d+)", query) or re.search(r"order\s+.*?(\d{4,})", query, re.IGNORECASE)
    messages: List[Dict[str, str]] = [{"role": "system", "content": PROMPT}]

    if chat_history:
        messages.extend(chat_history)
        messages.append({"role": "user", "content": query})
        
    iteration = 0
    order_data, profile_data, policy_data = None, None, None

    while iteration < max_iterations:
        iteration += 1

        # Step 1: Retrieve order if mentioned
        if has_order_number and order_data is None:
            order_id = has_order_number.group(1)
            order_data = _run_tool("get_order_details", {"order_id": order_id})
            steps.append({
                "step": len(steps)+1,
                "tool": "get_order_details",
                "input": {"order_id": order_id},
                "output": order_data
            })

            # Fetch customer profile if order succeeded
            if "error" not in order_data:
                customer_type = order_data.get("customer_type")
                profile_data = _run_tool("get_customer_profile", {"customer_type": customer_type})
                steps.append({
                    "step": len(steps)+1,
                    "tool": "get_customer_profile",
                    "input": {"customer_type": customer_type},
                    "output": profile_data
                })

        # Step 2: Retrieve policy if relevant
        if _needs_policy(query) and policy_data is None:
            policy_query = query
            if order_data and "customer_type" in order_data:
                policy_query = f"{order_data['customer_type']} return policy"
            policy_data = _run_tool("retrieve_policy", {"query": policy_query})
            steps.append({
                "step": len(steps)+1,
                "tool": "retrieve_policy",
                "input": {"query": policy_query},
                "output": policy_data
            })

        # Step 3: Prepare LLM input with tool results
        tool_results_summary = ""
        if order_data:
            tool_results_summary += f"Order Data:\n{json.dumps(order_data, indent=2)}\n"
        if profile_data:
            tool_results_summary += f"Customer Profile:\n{json.dumps(profile_data, indent=2)}\n"
        if policy_data:
            tool_results_summary += f"Policy:\n{policy_data.get('context','No policy found')}\n"

        messages.append({
            "role": "user",
            "content": f"{query}\n\nTool Results:\n{tool_results_summary}\nAnswer concisely based on the above information."
        })

        # Get LLM response
        response = llm.invoke(messages)
        answer = response.content

        # Stop loop for now; replace later with tool-calling check
        break

    return {
        "answer": answer,
        "steps": steps,
        "iterations": iteration
    }