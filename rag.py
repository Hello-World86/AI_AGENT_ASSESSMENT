policy_text = """
Standard customers have a 30-day return window and get a $20 credit for delays.
VIP customers have a 60-day return window and get full refunds for delays or damage.
"""

def retrieve_policy(customer_type):
    if customer_type == "VIP":
        return "VIP customers have a 60-day return window and get full refunds."
    else:
        return "Standard customers have a 30-day return window and get $20 credit."