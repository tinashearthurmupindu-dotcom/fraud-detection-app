"""
neo4j_connector.py
==================
FuneralGuard — Neo4j Graph Database Connector
----------------------------------------------
Drop this file alongside app_v27.py.

Add to requirements.txt:
    neo4j>=5.19.0

Usage (inside your Streamlit app):
    from neo4j_connector import Neo4jConnector, test_connection

Quick connection test:
    ok, msg = test_connection(uri, user, password, database)

Full driver usage:
    conn = Neo4jConnector(uri, user, password, database)
    df   = conn.run_query("MATCH (c:Claim) RETURN c.id AS id LIMIT 10")
    conn.close()
"""

from __future__ import annotations

import pandas as pd
from typing import Optional

# ── Try importing the neo4j driver ────────────────────────────────────
try:
    from neo4j import GraphDatabase, exceptions as neo4j_exc
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    neo4j_exc = None          # type: ignore


# ── Connection helper ─────────────────────────────────────────────────
class Neo4jConnector:
    """
    Thin wrapper around the official neo4j Python driver.

    Parameters
    ----------
    uri      : Bolt or Neo4j URI, e.g. "bolt://localhost:7687"
               or "neo4j+s://xxxxx.databases.neo4j.io" for AuraDB
    user     : Username (default "neo4j")
    password : Password
    database : Target database name (default "neo4j")
    """

    def __init__(
        self,
        uri: str,
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
    ) -> None:
        if not NEO4J_AVAILABLE:
            raise RuntimeError(
                "neo4j driver is not installed. "
                "Add `neo4j>=5.19.0` to requirements.txt and redeploy."
            )
        self.database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    # ── Verify connectivity ───────────────────────────────────────────
    def ping(self) -> bool:
        """Return True if the server is reachable."""
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    # ── Run a Cypher query and return a DataFrame ─────────────────────
    def run_query(self, cypher: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Execute *cypher* and return results as a pandas DataFrame.

        Parameters
        ----------
        cypher : Cypher query string
        params : Optional dict of query parameters

        Returns
        -------
        pd.DataFrame — empty DataFrame if no results
        """
        params = params or {}
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            records = [dict(r) for r in result]
        return pd.DataFrame(records) if records else pd.DataFrame()

    # ── Run a write transaction ───────────────────────────────────────
    def run_write(self, cypher: str, params: Optional[dict] = None) -> None:
        """Execute a write Cypher statement (CREATE / MERGE / SET …)."""
        params = params or {}
        with self._driver.session(database=self.database) as session:
            session.execute_write(lambda tx: tx.run(cypher, **params))

    # ── Fetch claim fraud network for a given claim_id ────────────────
    def fraud_network(self, claim_id: str, hops: int = 2) -> pd.DataFrame:
        """
        Return entities linked to *claim_id* within *hops* graph hops.
        Assumes the schema: (:Claim)-[:LINKED_TO]->(:Entity)
        """
        cypher = (
            f"MATCH (c:Claim {{id: $cid}})-[*1..{hops}]-(e:Entity) "
            "RETURN e.name AS entity, e.type AS type, "
            "e.risk AS risk, e.fraud_count AS fraud_count "
            "ORDER BY e.risk DESC LIMIT 100"
        )
        return self.run_query(cypher, {"cid": claim_id})

    # ── Close driver ─────────────────────────────────────────────────
    def close(self) -> None:
        """Close the underlying driver connection pool."""
        if self._driver:
            self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Standalone connection test (used by Streamlit UI) ─────────────────
def test_connection(
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
) -> tuple[bool, str]:
    """
    Try to open and ping a Neo4j connection.

    Returns
    -------
    (True, "Connected …")  on success
    (False, "Error …")     on failure
    """
    if not NEO4J_AVAILABLE:
        return False, (
            "neo4j driver not installed. "
            "Add `neo4j>=5.19.0` to requirements.txt and redeploy."
        )
    try:
        conn = Neo4jConnector(uri, user, password, database)
        if conn.ping():
            conn.close()
            return True, f"✅ Connected to {uri} (db: {database})"
        conn.close()
        return False, "Server reachable but ping failed — check credentials."
    except Exception as exc:
        return False, str(exc)
