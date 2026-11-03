-- Phase 05 additive schema. Never mutates legacy transcript rows.
CREATE TABLE IF NOT EXISTS ai_jobs (
 id uuid PRIMARY KEY, kind varchar(48) NOT NULL, meeting_id varchar(64),
 payload jsonb NOT NULL DEFAULT '{}'::jsonb, result jsonb,
 status varchar(16) NOT NULL DEFAULT 'queued'
 CHECK(status IN ('queued','running','completed','failed','cancelled')),
 attempts integer NOT NULL DEFAULT 0, max_attempts integer NOT NULL DEFAULT 2,
 priority integer NOT NULL DEFAULT 10, error text,
 created_at timestamptz NOT NULL DEFAULT now(),
 updated_at timestamptz NOT NULL DEFAULT now(), locked_at timestamptz,
 completed_at timestamptz
);
CREATE INDEX IF NOT EXISTS ai_jobs_claim_idx ON ai_jobs(status, priority, created_at);
CREATE INDEX IF NOT EXISTS ai_jobs_meeting_idx ON ai_jobs(meeting_id, created_at DESC);
CREATE TABLE IF NOT EXISTS ai_artifacts (
 id uuid PRIMARY KEY, meeting_id varchar(64), job_id uuid REFERENCES ai_jobs(id) ON DELETE SET NULL,
 kind varchar(48) NOT NULL, mode varchar(32), source_revision bigint,
 content jsonb NOT NULL, model_name text,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ai_artifacts_unique_job_idx ON ai_artifacts(job_id);
CREATE INDEX IF NOT EXISTS ai_artifacts_meeting_idx ON ai_artifacts(meeting_id, created_at DESC);
CREATE TABLE IF NOT EXISTS ai_actions (
 id uuid PRIMARY KEY, meeting_id varchar(64), artifact_id uuid REFERENCES ai_artifacts(id) ON DELETE CASCADE,
 title text NOT NULL, owner_name text, due_date date,
 status varchar(20) NOT NULL DEFAULT 'proposed'
 CHECK(status IN ('proposed','approved','in_progress','done','rejected')),
 evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
 revision integer NOT NULL DEFAULT 1,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS ai_decisions (
 id uuid PRIMARY KEY, meeting_id varchar(64), artifact_id uuid REFERENCES ai_artifacts(id) ON DELETE CASCADE,
 title text NOT NULL, status varchar(20) NOT NULL DEFAULT 'proposed'
 CHECK(status IN ('proposed','confirmed','rejected','superseded')),
 evidence jsonb NOT NULL DEFAULT '[]'::jsonb,
 revision integer NOT NULL DEFAULT 1,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS ai_media (
 id uuid PRIMARY KEY, file_name text NOT NULL, content_type text NOT NULL,
 local_path text NOT NULL UNIQUE, sha256 char(64) NOT NULL,
 size_bytes bigint NOT NULL, status varchar(20) NOT NULL DEFAULT 'uploaded',
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS ai_knowledge (
 id uuid PRIMARY KEY, source_kind varchar(16) NOT NULL,
 source_id varchar(64) NOT NULL, source_revision bigint NOT NULL DEFAULT 0,
 chunk_index integer NOT NULL, text_content text NOT NULL,
 evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
 embedding jsonb, embedding_model text,
 updated_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(source_kind, source_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS ai_knowledge_text_idx ON ai_knowledge USING gin(to_tsvector('simple', text_content));
CREATE INDEX IF NOT EXISTS ai_knowledge_source_idx ON ai_knowledge(source_kind, source_id);
CREATE TABLE IF NOT EXISTS ai_translations (
 id uuid PRIMARY KEY, meeting_id varchar(64) NOT NULL,
 segment_id bigint NOT NULL, source_text text NOT NULL,
 source_hash char(64) NOT NULL, target_language varchar(16) NOT NULL,
 translated_text text NOT NULL, model_name text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE(meeting_id, segment_id, source_hash, target_language)
);
