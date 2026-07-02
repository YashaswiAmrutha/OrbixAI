from graph.neo4j_client import neo4j_client

def save_contact(name,email):

    query = """
    MERGE (u:User {name:'Amrutha'})

    MERGE (p:Person {
        name:$name,
        email:$email
    })

    MERGE (u)-[:CONTACT_OF]->(p)
    """

    neo4j_client.run_query(
        query,
        {
            "name":name,
            "email":email
        }
    )