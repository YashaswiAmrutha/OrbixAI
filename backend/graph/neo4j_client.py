from neo4j import GraphDatabase

class Neo4jClient:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j","capstone_vapk")
        )

    def run_query(self, query, params=None):

        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

neo4j_client = Neo4jClient() 