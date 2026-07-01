from llm.ollama_client import generate_response
import json
import re

MEMORY_PROMPT = """
Extract contact information from the user message.

Return ONLY valid JSON.

Schema:

{
  "type":"contact",
  "name":"",
  "email":"",
  "phone":"",
  "relationship":""
}

Examples:

User:
My classmate Chandu's email is c4487776@gmail.com

Output:
{
  "type":"contact",
  "name":"Chandu",
  "email":"c4487776@gmail.com",
  "relationship":"classmate"
}

User:
My manager Divya's email is divya@gmail.com

Output:
{
  "type":"contact",
  "name":"Divya",
  "email":"divya@gmail.com",
  "relationship":"manager"
}

User:
{query}
"""


def extract_memory(text):

    prompt = MEMORY_PROMPT.replace("{query}", text)

    response = generate_response(prompt)

    print("\nRAW MEMORY RESPONSE:")
    print(response)

    try:
        match = re.search(r'\{[\s\S]*\}', response)

        if match:
            data = json.loads(match.group())

            print("PARSED MEMORY:", data)

            return data

    except Exception as e:
        print("JSON ERROR:", e)

    return None