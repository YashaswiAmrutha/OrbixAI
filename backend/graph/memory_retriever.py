from graph.neo4j_client import neo4j_client

def get_contact_email(name):

    query = """
    MATCH (:User)-[:KNOWS]->(c:Contact)
    WHERE toLower(c.name) CONTAINS toLower($name)
    RETURN c.email AS email
    """

    result = neo4j_client.run_query(
        query,
        {"name": name}
    )

    if result:
        return result[0]["email"]

    return None