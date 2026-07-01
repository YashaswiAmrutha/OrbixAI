from graph.neo4j_client import neo4j_client
def save_memory(memory):

    print("SAVE_MEMORY CALLED:", memory)

    if memory.get("type") != "contact":
        return

    if not memory.get("name"):
        print("SKIPPING CONTACT - NO NAME FOUND")
        return

    query = """
    MERGE (u:User {name:'Amrutha'})

    MERGE (c:Contact {name:$name})

    SET c.email = $email
    SET c.phone = $phone
    SET c.relationship = $relationship

    MERGE (u)-[:KNOWS]->(c)
    """

    neo4j_client.run_query(
        query,
        {
            "name": memory.get("name"),
            "email": memory.get("email"),
            "phone": memory.get("phone"),
            "relationship": memory.get("relationship")
        }
    )

    print("CONTACT SAVED")