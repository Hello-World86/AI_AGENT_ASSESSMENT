# tools.py

# Mock CRM order database
orders = {
    8892: {
        "order_id": 8892,
        "customer_type": "VIP",
        "status": "Delivered",
        "days_since_delivery": 45
    },
    9910: {
        "order_id": 9910,
        "customer_type": "Standard",
        "status": "Processing",
        "days_since_delivery": 2
    }
}

# Customer tier information
customer_profiles = {
    "VIP": {
        "tier": "VIP",
        "return_window": 60,
        "benefits": "Full refunds for delays or damage."
    },
    "Standard": {
        "tier": "Standard",
        "return_window": 30,
        "benefits": "$20 credit for delays."
    }
}


def get_order_details(order_id: int):
    return orders.get(order_id, {"error": "Order not found"})


def get_customer_profile(customer_type: str):
    return customer_profiles.get(customer_type, {"error": "Customer type not found"})