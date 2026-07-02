from neo4j_client import neo4j_client

query = """
MERGE (u:User {name:'Amrutha'})

MERGE (p:Person {name:'John'})
MERGE (p)-[:COLLEAGUE_OF]->(u)

MERGE (proj:Project {name:'OrbixAI'})
MERGE (u)-[:WORKS_ON]->(proj)
"""

neo4j_client.run_query(query)