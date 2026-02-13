def build_prompt(user_message: str) -> str:
    return f"""
You are Orbii, a friendly and helpful desktop AI assistant.
Respond conversationally and clearly.

User: {user_message}
Orbii:
"""
