import json
import os
import typing as t

import psycopg2
from fastapi import FastAPI, Query
from psycopg2.extras import Json

app = FastAPI()

DB_URI = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"
)


def build_filter_query(filters: dict[str, str]) -> tuple[str, tuple]:
    """Build a PostgreSQL JSONB query from filters"""
    if not filters:
        return "TRUE", ()

    conditions = []
    params = []

    for key, value in filters.items():
        conditions.append("data->>%s = %s")
        params.extend([key, value])

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    return where_clause, tuple(params)


@app.post("/push")
def push_data(rows: list[dict[str, t.Any]]):
    """Insert new rows into the database"""
    with psycopg2.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    "INSERT INTO mlperf_results (data) VALUES (%s)",
                    (Json(row),),
                )
    return {"status": "success"}


@app.get("/list")
def list_ids(filters: str | None = Query(None)) -> list[int]:
    """Get list of IDs matching filters"""
    filter_dict = json.loads(filters) if filters else {}
    where_clause, params = build_filter_query(filter_dict)
    query = f"SELECT id FROM mlperf_results WHERE {where_clause}"

    with psycopg2.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            results = [row[0] for row in cur.fetchall()]

    return results


@app.get("/pull")
def pull_data(filters: str | None = Query(None)) -> list[dict[str, t.Any]]:
    """Get full data matching filters"""
    filter_dict = json.loads(filters) if filters else {}
    where_clause, params = build_filter_query(filter_dict)
    query = f"SELECT data FROM mlperf_results WHERE {where_clause}"

    with psycopg2.connect(DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            results = [row[0] for row in cur.fetchall()]

    return results
