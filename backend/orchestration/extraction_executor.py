"""
Phase 3: Extraction Executor — Background task parallelization

Handles parallel execution of:
1. Fact extraction from conversation turns (chat responses)
2. Trip extraction and Neo4j storage (travel module)
3. Action extraction and entity linking (action module)
4. Cache updates to SQLite

Uses asyncio.gather() for parallel processing and returns to caller immediately.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)


class ExtractionExecutor:
    """Executes extraction tasks in parallel (Phase 3)."""

    def __init__(self):
        self.tasks_executed = 0
        self.tasks_failed = 0

    async def execute_tasks(self, extraction_tasks: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
        """
        Execute extraction tasks in parallel.
        
        Returns {success_count, failed_count, errors}.
        Does NOT block caller — returns immediately after queueing.
        """
        if not extraction_tasks:
            return {"success": 0, "failed": 0, "errors": []}

        logger.info(f"Executing {len(extraction_tasks)} extraction tasks for session {session_id}")

        # Prepare coroutines for parallel execution
        coros = []
        for task in extraction_tasks:
            task_type = task.get("type")
            
            if task_type == "extract_from_turn":
                coros.append(self._extract_from_turn(task))
            elif task_type == "extract_trip_to_neo4j":
                coros.append(self._extract_trip_to_neo4j(task))
            elif task_type == "extract_action":
                coros.append(self._extract_action(task))
            else:
                logger.warning(f"Unknown extraction task type: {task_type}")

        # Execute all tasks in parallel
        if not coros:
            return {"success": 0, "failed": 0, "errors": []}

        try:
            results = await asyncio.gather(*coros, return_exceptions=True)
            
            success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
            failed_count = len(results) - success_count
            errors = [r.get("error") for r in results if isinstance(r, dict) and r.get("error")]
            
            logger.info(f"Extraction tasks completed: {success_count} success, {failed_count} failed")
            return {"success": success_count, "failed": failed_count, "errors": errors}
            
        except Exception as e:
            logger.error(f"Error executing extraction tasks: {e}", exc_info=True)
            return {"success": 0, "failed": len(extraction_tasks), "errors": [str(e)]}

    async def _extract_from_turn(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract durable facts from the user's turn and persist them to Neo4j.

        This is now the single write path for chat memory (the old flush-barrier /
        SQLite-queue drain has been removed). It reuses the proven extraction engine
        (heuristic gate -> structured-output LLM -> resolve/reconcile/upsert) so writes
        land in the same :Entity/:Fact/:Memory graph the agent reads via recall().

        Only the user's turn is mined (matching the original design — the assistant's
        text is not treated as a source of durable facts).
        """
        try:
            user_text = task.get("user_text", "")
            session_id = task.get("session_id")

            from mcp_host import extractor

            def _run() -> int:
                # Cheap gate first; skip questions/greetings/chit-chat without an LLM call.
                if not extractor.is_memory_worthy(user_text):
                    return 0
                result = extractor.extract(user_text)          # structured-output LLM
                return extractor._persist(result, source="langgraph")  # -> Neo4j (idempotent MERGE)

            writes = await asyncio.to_thread(_run)
            logger.info(f"extract_from_turn: {writes} write(s) to Neo4j (session {session_id})")
            return {"success": True, "writes": writes}

        except Exception as e:
            # Fire-and-forget: on failure (e.g. Neo4j down) the fact is lost — there is no
            # durable retry queue anymore. Logged so it's visible.
            logger.error(f"Error in _extract_from_turn: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _extract_trip_to_neo4j(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and store trip information to Neo4j.
        Creates :Trip node with :Flight, :Stay, :Location children.
        """
        try:
            from_city = task.get("from_city")
            to_city = task.get("to_city")
            check_in = task.get("check_in")
            check_out = task.get("check_out")
            num_adults = task.get("num_adults", 1)
            itinerary = task.get("itinerary", "")
            flights = task.get("flights", [])
            hotels = task.get("hotels", [])
            attractions = task.get("attractions", [])
            session_id = task.get("session_id")

            logger.info(f"Storing trip to Neo4j: {from_city} → {to_city}")

            from graph.connection import get_driver
            driver = get_driver()

            # Create :Trip node in Neo4j
            trip_id = str(uuid4())[:8]
            with driver.session() as session:
                # Merge trip node (idempotent)
                session.run(
                    """
                    MERGE (t:Trip {id: $trip_id})
                    ON CREATE SET 
                        t.from_city = $from_city,
                        t.to_city = $to_city,
                        t.check_in = $check_in,
                        t.check_out = $check_out,
                        t.num_adults = $num_adults,
                        t.created_at = datetime()
                    ON MATCH SET
                        t.updated_at = datetime()
                    RETURN t
                    """,
                    {
                        "trip_id": trip_id,
                        "from_city": from_city,
                        "to_city": to_city,
                        "check_in": check_in,
                        "check_out": check_out,
                        "num_adults": num_adults,
                    }
                )

                # Create :Location node for destination
                session.run(
                    """
                    MERGE (l:Location {name: $city, type: 'city'})
                    ON CREATE SET l.created_at = datetime()
                    WITH l
                    MATCH (t:Trip {id: $trip_id})
                    MERGE (t)-[:DESTINATION]->(l)
                    """,
                    {"city": to_city, "trip_id": trip_id}
                )

                # Create :Flight nodes
                for idx, flight in enumerate(flights[:3]):  # Limit to top 3
                    session.run(
                        """
                        MERGE (f:Flight {
                            trip_id: $trip_id,
                            price: $price,
                            departure: $departure,
                            arrival: $arrival
                        })
                        ON CREATE SET 
                            f.duration = $duration,
                            f.stops = $stops,
                            f.created_at = datetime()
                        WITH f
                        MATCH (t:Trip {id: $trip_id})
                        MERGE (t)-[:FLIGHT]->(f)
                        """,
                        {
                            "trip_id": trip_id,
                            "price": flight.get("price"),
                            "departure": flight.get("departure"),
                            "arrival": flight.get("arrival"),
                            "duration": flight.get("duration"),
                            "stops": flight.get("stops"),
                        }
                    )

                # Create :Stay nodes
                for idx, hotel in enumerate(hotels[:3]):  # Limit to top 3
                    session.run(
                        """
                        MERGE (s:Stay {
                            trip_id: $trip_id,
                            name: $name,
                            price: $price
                        })
                        ON CREATE SET 
                            s.room_type = $room_type,
                            s.created_at = datetime()
                        WITH s
                        MATCH (t:Trip {id: $trip_id})
                        MERGE (t)-[:STAY]->(s)
                        """,
                        {
                            "trip_id": trip_id,
                            "name": hotel.get("name"),
                            "price": hotel.get("price"),
                            "room_type": hotel.get("room_type"),
                        }
                    )

            logger.info(f"Trip {trip_id} stored to Neo4j with {len(flights)} flights, {len(hotels)} hotels")
            return {"success": True, "trip_id": trip_id, "stored": "Neo4j"}

        except Exception as e:
            logger.error(f"Error in _extract_trip_to_neo4j: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    async def _extract_action(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract action metadata and link to relevant entities.
        For emails: extract recipient, subject, date
        For meetings: extract attendees, date, time
        """
        try:
            action = task.get("action")
            parameters = task.get("parameters", {})
            session_id = task.get("session_id")

            logger.info(f"Extracting action '{action}' to Neo4j")

            from graph.connection import get_driver
            driver = get_driver()

            with driver.session() as session:
                if action == "send_email":
                    # Create email action node
                    session.run(
                        """
                        MERGE (e:Email {
                            recipient: $recipient,
                            subject: $subject,
                            created_at: datetime()
                        })
                        """,
                        {
                            "recipient": parameters.get("recipient_email"),
                            "subject": parameters.get("subject", "[No Subject]"),
                        }
                    )

                elif action in ("create_meeting", "meeting_and_email"):
                    # Create meeting node
                    session.run(
                        """
                        MERGE (m:Meeting {
                            title: $title,
                            date: $date,
                            attendee: $attendee,
                            created_at: datetime()
                        })
                        """,
                        {
                            "title": parameters.get("event_title"),
                            "date": parameters.get("date", ""),
                            "attendee": parameters.get("attendee_email"),
                        }
                    )

            logger.info(f"Action '{action}' extracted to Neo4j")
            return {"success": True, "action": action}

        except Exception as e:
            logger.error(f"Error in _extract_action: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


# Singleton executor
_executor = None


def get_executor() -> ExtractionExecutor:
    """Get or create the extraction executor."""
    global _executor
    if _executor is None:
        _executor = ExtractionExecutor()
    return _executor
