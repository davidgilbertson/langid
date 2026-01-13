-- SQLite schema for data/dataset.sqlite produced by tools/export_sqlite.py

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS samples (
  content TEXT NOT NULL,
  language TEXT NOT NULL,
  extension TEXT NOT NULL,
  length_chars INTEGER NOT NULL,
  annotations_json TEXT NOT NULL,
  linguist TEXT NOT NULL,
  path TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS languages (
  name TEXT PRIMARY KEY,
  linguist_primary TEXT,
  rosetta_code_primary TEXT
);
