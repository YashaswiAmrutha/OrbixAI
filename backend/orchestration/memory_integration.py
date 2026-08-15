"""
Phase 5: Memory Integration — Neo4j schema, entity linking, and deduplication

Implements:
1. Trip node structure with relationships
2. Entity linking (emails to trips, contacts to organizations)
3. Merge-based deduplication (idempotent writes)
4. Profile integration (user context for all queries)
"""

import logging
from typing import Dict, List, Any, Optional
from graph.connection import get_driver

logger = logging.getLogger(__name__)


class MemoryIntegration:
    """Neo4j memory integration for long-term knowledge graph."""

    def __init__(self):
        self.driver = get_driver()

    # ======================================================================
    # USER PROFILE INITIALIZATION
    # ======================================================================

    def ensure_user_profile(self, user_data: Dict[str, Any]) -> bool:
        """
        Ensure :User singleton exists with base properties.
        Called once on startup or when user profile changes.
        
        User data: {name, email, city, birthday, preferences}
        """
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (u:User {id: 'user:self'})
                    ON CREATE SET 
                        u.name = $name,
                        u.email = $email,
                        u.city = $city,
                        u.created_at = datetime()
                    ON MATCH SET
                        u.name = COALESCE($name, u.name),
                        u.email = COALESCE($email, u.email),
                        u.city = COALESCE($city, u.city),
                        u.updated_at = datetime()
                    RETURN u
                    """,
                    {
                        "name": user_data.get("name"),
                        "email": user_data.get("email"),
                        "city": user_data.get("city"),
                    }
                )
                logger.info("User profile ensured in Neo4j")
                return True
        except Exception as e:
            logger.error(f"Error ensuring user profile: {e}")
            return False

    # ======================================================================
    # TRIP SCHEMA & STORAGE
    # ======================================================================

    def store_trip(self, trip_data: Dict[str, Any]) -> Optional[str]:
        """
        Store complete trip to Neo4j with schema:
        
        :User -[:PLANS]-> :Trip -[:DESTINATION]-> :Location
                            ├─ -[:FLIGHT]-> :Flight
                            ├─ -[:STAY]-> :Stay
                            └─ -[:VISIT]-> :Attraction
        """
        try:
            trip_id = trip_data.get("id") or f"trip:{trip_data.get('to_city')}:{trip_data.get('check_in')}"

            with self.driver.session() as session:
                # 1. Create Trip node
                session.run(
                    """
                    MERGE (t:Trip:Entity {id: $trip_id})
                    ON CREATE SET
                        t.key = $trip_id,
                        t.name = $to_city,
                        t.from_city = $from_city,
                        t.to_city = $to_city,
                        t.check_in = $check_in,
                        t.check_out = $check_out,
                        t.num_adults = $num_adults,
                        t.num_nights = $num_nights,
                        t.status = 'planned',
                        t.created_at = datetime()
                    ON MATCH SET
                        t.updated_at = datetime()
                    WITH t
                    MATCH (u:User {id: 'user:self'})
                    MERGE (u)-[:PLANS {created_at: datetime()}]->(t)
                    """,
                    {
                        "trip_id": trip_id,
                        "from_city": trip_data.get("from_city", "Unknown"),
                        "to_city": trip_data.get("to_city"),
                        "check_in": trip_data.get("check_in"),
                        "check_out": trip_data.get("check_out"),
                        "num_adults": trip_data.get("num_adults", 1),
                        "num_nights": trip_data.get("num_nights", 1),
                    }
                )

                # 2. Create Destination Location
                to_city = trip_data.get("to_city")
                session.run(
                    """
                    MERGE (loc:Location:Entity {name: $city, type: 'city'})
                    ON CREATE SET
                        loc.key = 'location:' + toLower($city),
                        loc.country = $country,
                        loc.created_at = datetime()
                    WITH loc
                    MATCH (t:Trip {id: $trip_id})
                    MERGE (t)-[:DESTINATION {visited: false}]->(loc)
                    """,
                    {
                        "trip_id": trip_id,
                        "city": to_city,
                        "country": trip_data.get("country"),
                    }
                )

                # 3. Create Flight options
                flights = trip_data.get("flights", [])
                for idx, flight in enumerate(flights[:5]):
                    session.run(
                        """
                        MERGE (f:Flight {
                            trip_id: $trip_id,
                            price: $price,
                            departure: $departure
                        })
                        ON CREATE SET 
                            f.arrival = $arrival,
                            f.duration = $duration,
                            f.stops = $stops,
                            f.currency = $currency,
                            f.rank = $rank,
                            f.created_at = datetime()
                        WITH f
                        MATCH (t:Trip {id: $trip_id})
                        MERGE (t)-[:FLIGHT {option_num: $rank}]->(f)
                        """,
                        {
                            "trip_id": trip_id,
                            "price": str(flight.get("price", "")),
                            "departure": flight.get("departure"),
                            "arrival": flight.get("arrival"),
                            "duration": flight.get("duration"),
                            "stops": flight.get("stops", 0),
                            "currency": flight.get("currency", ""),
                            "rank": idx + 1,
                        }
                    )

                # 4. Create Hotel options
                hotels = trip_data.get("hotels", [])
                for idx, hotel in enumerate(hotels[:5]):
                    session.run(
                        """
                        MERGE (h:Stay {
                            trip_id: $trip_id,
                            name: $name,
                            price: $price
                        })
                        ON CREATE SET 
                            h.room_type = $room_type,
                            h.currency = $currency,
                            h.rank = $rank,
                            h.created_at = datetime()
                        WITH h
                        MATCH (t:Trip {id: $trip_id})
                        MERGE (t)-[:STAY {option_num: $rank}]->(h)
                        """,
                        {
                            "trip_id": trip_id,
                            "name": hotel.get("name", ""),
                            "price": str(hotel.get("price", "")),
                            "room_type": hotel.get("room_type"),
                            "currency": hotel.get("currency", ""),
                            "rank": idx + 1,
                        }
                    )

                # 5. Create Attractions
                attractions = trip_data.get("attractions", [])
                for idx, attraction in enumerate(attractions[:20]):
                    session.run(
                        """
                        MERGE (a:Attraction {
                            name: $name,
                            city: $city,
                            category: $category
                        })
                        ON CREATE SET 
                            a.lat = $lat,
                            a.lon = $lon,
                            a.created_at = datetime()
                        WITH a
                        MATCH (t:Trip {id: $trip_id})
                        MERGE (t)-[:VISIT {priority: $priority}]->(a)
                        """,
                        {
                            "trip_id": trip_id,
                            "name": attraction.get("name"),
                            "city": to_city,
                            "category": attraction.get("category"),
                            "lat": attraction.get("lat"),
                            "lon": attraction.get("lon"),
                            "priority": idx + 1,
                        }
                    )

            logger.info(f"Trip {trip_id} stored with flights, hotels, attractions")
            return trip_id

        except Exception as e:
            logger.error(f"Error storing trip: {e}", exc_info=True)
            return None

    # ======================================================================
    # EMAIL & CONTACT LINKING
    # ======================================================================

    def link_email_to_contact(self, email: str, name: Optional[str] = None) -> bool:
        """
        Ensure email address is linked to a :Person contact.
        Merges if person already exists.
        """
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (p:Person {email: $email})
                    ON CREATE SET 
                        p.name = $name,
                        p.created_at = datetime()
                    ON MATCH SET
                        p.name = COALESCE($name, p.name)
                    WITH p
                    MATCH (u:User {id: 'user:self'})
                    MERGE (u)-[:KNOWS {since: datetime()}]->(p)
                    """,
                    {"email": email, "name": name or email}
                )
                logger.info(f"Email {email} linked to contact")
                return True
        except Exception as e:
            logger.error(f"Error linking email: {e}")
            return False

    def link_email_to_trip(self, email: str, trip_id: str) -> bool:
        """
        Link an email recipient to a trip.
        Useful for tracking "who I'm inviting to the trip".
        """
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (p:Person {email: $email})
                    WITH p
                    MATCH (t:Trip {id: $trip_id})
                    MERGE (p)-[:INVITED_TO {date: datetime()}]->(t)
                    """,
                    {"email": email, "trip_id": trip_id}
                )
                logger.info(f"Email {email} linked to trip {trip_id}")
                return True
        except Exception as e:
            logger.error(f"Error linking email to trip: {e}")
            return False

    # ======================================================================
    # QUERY & RECALL HELPERS
    # ======================================================================

    def get_recent_trips(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent trip plans from Neo4j."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (u:User {id: 'user:self'})-[:PLANS]->(t:Trip)
                    MATCH (t)-[:DESTINATION]->(loc:Location)
                    RETURN t.id as trip_id, t.to_city as destination, 
                           t.check_in as check_in, t.status as status,
                           t.created_at as created_at
                    ORDER BY t.created_at DESC
                    LIMIT $limit
                    """,
                    {"limit": limit}
                )
                trips = [dict(record) for record in result]
                logger.info(f"Retrieved {len(trips)} recent trips")
                return trips
        except Exception as e:
            logger.error(f"Error fetching recent trips: {e}")
            return []

    def get_contacts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch contact list from Neo4j."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (u:User {id: 'user:self'})-[:KNOWS]->(p:Person)
                    RETURN p.name as name, p.email as email, p.created_at as created_at
                    ORDER BY p.created_at DESC
                    LIMIT $limit
                    """,
                    {"limit": limit}
                )
                contacts = [dict(record) for record in result]
                logger.info(f"Retrieved {len(contacts)} contacts")
                return contacts
        except Exception as e:
            logger.error(f"Error fetching contacts: {e}")
            return []

    def get_preferences(self) -> Dict[str, Any]:
        """Fetch user preferences from Neo4j."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (u:User {id: 'user:self'})-[:PREFERS]->(p:Preference)
                    RETURN p.key as key, p.value as value
                    """
                )
                prefs = {record["key"]: record["value"] for record in result}
                logger.info(f"Retrieved {len(prefs)} preferences")
                return prefs
        except Exception as e:
            logger.error(f"Error fetching preferences: {e}")
            return {}

    # ======================================================================
    # DEDUPLICATION (MERGE Protocol)
    # ======================================================================

    def deduplicate_trip(self, trip_data: Dict[str, Any]) -> Optional[str]:
        """
        Merge trip data intelligently to avoid duplicates.
        Uses destination + check_in as unique key.
        """
        try:
            to_city = trip_data.get("to_city")
            check_in = trip_data.get("check_in")

            if not to_city or not check_in:
                logger.warning("Cannot deduplicate trip without to_city and check_in")
                return None

            with self.driver.session() as session:
                # Check if trip already exists
                result = session.run(
                    """
                    MATCH (t:Trip)
                    WHERE t.to_city = $to_city AND t.check_in = $check_in
                    RETURN t.id as trip_id
                    LIMIT 1
                    """,
                    {"to_city": to_city, "check_in": check_in}
                )

                existing = result.single()
                if existing:
                    trip_id = existing["trip_id"]
                    logger.info(f"Trip already exists: {trip_id}, merging data")
                    # Update existing trip instead of creating new one
                    session.run(
                        """
                        MATCH (t:Trip {id: $trip_id})
                        SET t.updated_at = datetime()
                        RETURN t
                        """,
                        {"trip_id": trip_id}
                    )
                    return trip_id
                else:
                    # New trip
                    return self.store_trip(trip_data)

        except Exception as e:
            logger.error(f"Error deduplicating trip: {e}")
            return None


# Singleton
_memory_integration = None


def get_memory_integration() -> MemoryIntegration:
    """Get or create the memory integration."""
    global _memory_integration
    if _memory_integration is None:
        _memory_integration = MemoryIntegration()
    return _memory_integration
