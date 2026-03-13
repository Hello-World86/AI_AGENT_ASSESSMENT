import os
import json
import re 
from typing import Any 

from tools import get_order_details, get_customer_profile
from rag import retrieve_policy
from model import get_model

TOOL_REGISTRY = {
    "get_oder_details": get_order_details,
    "get_customer_profile": get_customer_profile
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
                    "description": "The order ID mentioned by the user (e.g., '2022' )."
                }
            },
            "required": ["order_id"]
        },
    },
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
#system prompt for the agent(master prompt)
PROMPT = """
You are a concise conversational AI agent that answers customers queries.
Your job is to answer customer questions about orders, delivery status and company policies.
You have access to tools that allow you to retrieve structured data from a CRM tool and retrieve company policies from a RAG.
FOllow these rules strictly. Provide answer in plain facts and no elaborations.

You have access to the following tools:

1. get_order_details
Use this tool whenever theh user references an order number(e.g. "Order #3031", "#3031","order 3031").

This tool returns:
- order id
- customer type(VIP or Standard)
- order status
- number of days since delivery 

2. get_customer_profile
Use this tool when you need to know the customer type (VIP or Standard) to determine eligibility for certain policies or benefits. 
This tool returns:
- customer type (VIP or Standard)   

3. retrieve_policy
Use this tool when the user asks about policies such as:
- return window
- refund policy
- delay compensation
- damaged items

Always follow this reasoning approach:
Step 1: Understand the user's question.
Step 2: Determine whether you need to retrieve:
- structured data (use get_order_details)
- policy information(use retrieve_policy)
- or both.
Step 3: If an order number is mentioned, call get_order_details first.
Step 4: If the question involves return eligibility or refunds, retrieve the relevant policy using retrieve_policy.
Step 5: Combine the information from the order data and policy to produce a clear answer.

IMPORTANT RULES
Never guess order information.
Always use tools when order data is required. 
Donot guess or invent policies.

If the user asks about the return eligibility, you must check the policy rules only.
"""
POLICY_KEYWORDS = [
    "policy", "return", "refund", "compensation", "delay", "damage", 
    "eligibility", "return window", "refund policy", "delay compensation",
    "damage claim", "eligible for return", "eligible for refund","eligible for compensation",
    "return policy", "refund eligibility", "compensation eligibility",
    "based on the policy", "according to the policy", "what is the policy for", "what's the policy for",
    "can I return", "can I get a refund", "can I get compensation", "how long is the return window", "what is the return window",
]

def _needs_policy(query:str)->bool:
    """"Determine if the user's query is related to company policies based on keyword matching."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in POLICY_KEYWORDS)

def _run_tool(tool_name:str, parameters:dict)->Any:
    """Execute the specified tool with the given parameters."""
    if tool_name == "retrieve_policy":
        result = retrieve_policy(**tool_input)
    elif tool_name in TOOL_REGISTRY:
        result = TOOL_REGISTRY[tool_name](**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result, indent=2)

def run_agent(
        query:str,
        max_iterations:int = 15,
        chat_history:list = [dict]| None = None,
        )->dict:
    """Main function for processing user queries and generating responses using tools."""
    llm = get_llm()

    steps: list[dict]= []
    has_order_number = re.search(r"#\s*(\d+)", query) or re.search(r"order\s+.*?(\d{4,})", query, re.IGNORECASE)
   
   messages = [{"role": "system", "content": PROMPT}]

    if chat_history:
        messeges.extend(chat_history)
        messeges.append({"role": "user", "content": query})

    if has_order_number:
        order_id = has_order_number.group(1)
        # Step 1: Call get_order_details tool when an order number is mentioned
        order_data = get_order_details(order_id)
        steps.append({
            "step": 1,
            "tool": "get_order_details",
            "input": {"order_id": order_id},
            "output": order_data
        })

        if "error" not in order_data and _needs_policy(query):
            # Step 2: If the question is about policies, call retrieve_policy with relevant query
            policy_query = f"{order_data.get('customer_type', '')} return policy"
            policy_data = retrieve_policy(f"{policy_query} return window")
            steps.append({
                "step": 2,
                "tool": "retrieve_policy",
                "input": {"query":  f"{policy_query} return window"},
                "output": policy_data
            })
            messeges.append({
                "role": "user", "content":(
                   f"{query}\n\n"
                    f"Tool Results:\n"
                    f"1. Order Data: {json.dumps(order_data, indent=2)}\n"
                    f"2. Policy: {policy_data.get('context', 'No policy found')}\n\n"
                    f"Answer the user's question based on the above information. Provide a concise and factual response without elaboration."
                ),
            })
            elif "error" not in order_data:
            messeges.append({
                    "role": "user", "content":(
                        f"{query}\n\n"
                        f"Tool Results:\n"
                        f"1. Order Data: {json.dumps(order_data, indent=2)}\n\n"
                        f"Answer the user's question based on the above information. Provide a concise and factual response without elaboration."
                    ),
                })
            else:
            # If no order number is mentioned but the query is policy-related, call retrieve_policy directly
            messeges.append({
                "role": "user", "content":(
                    f"{query}\n\n"
                    f"Tool Results:\n"
                    f"{json.dumps(policy_data, indent=2)}\n\n"
                    f"1. Policy: {policy_data.get('context', 'No policy found')}\n\n"
                    f"Respond that the order information is not available."
                ),
            })
    elif _needs_policy(query):
        policy_data = retrieve_policy(query)
        steps.append({
            "step": 1,
            "tool": "retrieve_policy",
            "input": {"query": query},
            "output": policy_data
        })
        messeges.append({
            "role": "user", "content":(
                f"{query}\n\n"
                f"Tool Results:\n"
                f"1. Policy: {policy_data.get('context', 'No policy found')}\n\n"
                f"Answer the user's question based on the above information. Provide a concise and factual response without elaboration."
            ),
        })
    else:
    # For queries that don't mention an order number or policy, respond directly without calling tools
        messeges.append({
            "role": "user", "content": query,
        })  

        #get response from the model llm
    response = llm.invoke(messeges)
    answer = response.content
    return {
        "answer": answer,
        "steps": steps,
        "iterations": len(steps) if steps else 1,
    }
