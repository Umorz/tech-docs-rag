PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- Key/value metadata (e.g. corpus_root absolute path, schema_version, etc.)
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Document registry (Step A populates this table)
CREATE TABLE IF NOT EXISTS documents (
  doc_id      INTEGER PRIMARY KEY,
  rel_path    TEXT NOT NULL UNIQUE, -- path relative to corpus_root, stored with forward slashes
  mtime_ns    INTEGER NOT NULL,
  size_bytes  INTEGER NOT NULL,
  status      TEXT NOT NULL DEFAULT 'new'
              CHECK(status IN ('new','changed','unchanged','indexed','error')),
  last_error  TEXT,
  indexed_at  INTEGER,              -- unix epoch seconds (set when pages/lines are indexed successfully)
  created_at  INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
  updated_at  INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- Page-level text storage (Step B populates this table)
CREATE TABLE IF NOT EXISTS pages (
  page_id    INTEGER PRIMARY KEY,
  doc_id     INTEGER NOT NULL,
  page_no    INTEGER NOT NULL,       -- 1-based page number for user-facing citations
  text       TEXT NOT NULL,
  is_ocr     INTEGER NOT NULL DEFAULT 0 CHECK(is_ocr IN (0,1)),
  text_len   INTEGER NOT NULL,
  text_hash  TEXT,                   -- optional: for debugging/dedup/incremental updates
  created_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
  FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_pages_doc_page ON pages(doc_id, page_no);

-- Line-level storage for precise quoting (Step B populates this table)
CREATE TABLE IF NOT EXISTS lines (
  line_id   INTEGER PRIMARY KEY,
  page_id   INTEGER NOT NULL,
  line_no   INTEGER NOT NULL,        -- 1-based line number within the page text
  text      TEXT NOT NULL,
  text_len  INTEGER NOT NULL,
  FOREIGN KEY(page_id) REFERENCES pages(page_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lines_page_line ON lines(page_id, line_no);

-- Corpus lexicon for Query-time typo correction (Step 2).
-- We store only "ordinary words" (letters-only) extracted from the indexed corpus.
-- This avoids "correcting" part numbers / identifiers like LM2596, 3.3V, I/O, etc.
CREATE TABLE IF NOT EXISTS lexicon (
  term TEXT PRIMARY KEY,
  freq INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lexicon_freq ON lexicon(freq);

-- Chunk storage for semantic search (Step 5: embeddings).
--
-- We store quote-friendly chunks linked to a page and (optionally) a line range.
-- Embeddings themselves are stored separately so you can keep multiple models.
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id      INTEGER PRIMARY KEY,
  page_id       INTEGER NOT NULL,
  start_line_no INTEGER, -- inclusive, 1-based within the page (nullable for page-level chunks)
  end_line_no   INTEGER, -- inclusive, 1-based within the page (nullable for page-level chunks)
  text          TEXT NOT NULL,
  text_len      INTEGER NOT NULL,
  created_at    INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
  FOREIGN KEY(page_id) REFERENCES pages(page_id) ON DELETE CASCADE
);

-- Helpful indices for chunk lookup and stable citations
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page_lines ON chunks(page_id, start_line_no, end_line_no);
CREATE INDEX IF NOT EXISTS idx_chunks_text_len ON chunks(text_len);

-- Helpful view to map chunk -> (document path, page number) for citations/debugging
CREATE VIEW IF NOT EXISTS v_chunk_sources AS
SELECT
  c.chunk_id,
  d.doc_id,
  d.rel_path,
  p.page_id,
  p.page_no,
  c.start_line_no,
  c.end_line_no,
  p.is_ocr,
  c.text_len
FROM chunks c
JOIN pages p ON p.page_id = c.page_id
JOIN documents d ON d.doc_id = p.doc_id;

-- Embeddings for chunks (supports multiple models via UNIQUE(chunk_id, model)).
-- `vector` is stored as a BLOB (e.g., float32 bytes). The vector index (FAISS/HNSW/etc.)
-- can live outside SQLite and reference chunks by chunk_id.
CREATE TABLE IF NOT EXISTS chunk_embeddings (
  embed_id    INTEGER PRIMARY KEY,
  chunk_id    INTEGER NOT NULL,
  model       TEXT NOT NULL,
  dim         INTEGER NOT NULL,
  vector      BLOB NOT NULL,
  created_at  INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
  FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  UNIQUE(chunk_id, model)
);

-- Helpful indices for model-scoped retrieval and filtering
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model ON chunk_embeddings(model);
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model_dim ON chunk_embeddings(model, dim);
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_chunk ON chunk_embeddings(chunk_id);

-- Full-text search (FTS5) indices
-- Note: these are "external content" FTS tables; triggers keep them in sync.
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
  text,
  content='pages',
  content_rowid='page_id',
  tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS lines_fts USING fts5(
  text,
  content='lines',
  content_rowid='line_id',
  tokenize='unicode61 remove_diacritics 2'
);

-- Sync triggers for pages -> pages_fts
CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
  INSERT INTO pages_fts(rowid, text) VALUES (new.page_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE OF text ON pages BEGIN
  INSERT INTO pages_fts(pages_fts, rowid, text) VALUES ('delete', old.page_id, old.text);
  INSERT INTO pages_fts(rowid, text) VALUES (new.page_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
  INSERT INTO pages_fts(pages_fts, rowid, text) VALUES ('delete', old.page_id, old.text);
END;

-- Sync triggers for lines -> lines_fts
CREATE TRIGGER IF NOT EXISTS lines_ai AFTER INSERT ON lines BEGIN
  INSERT INTO lines_fts(rowid, text) VALUES (new.line_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS lines_au AFTER UPDATE OF text ON lines BEGIN
  INSERT INTO lines_fts(lines_fts, rowid, text) VALUES ('delete', old.line_id, old.text);
  INSERT INTO lines_fts(rowid, text) VALUES (new.line_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS lines_ad AFTER DELETE ON lines BEGIN
  INSERT INTO lines_fts(lines_fts, rowid, text) VALUES ('delete', old.line_id, old.text);
END;

-- Helpful view for debugging sources (doc path + page)
CREATE VIEW IF NOT EXISTS v_page_sources AS
SELECT
  p.page_id,
  d.doc_id,
  d.rel_path,
  p.page_no,
  p.is_ocr,
  p.text_len
FROM pages p
JOIN documents d ON d.doc_id = p.doc_id;
