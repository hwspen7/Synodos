CREATE TABLE IF NOT EXISTS meeting_records (
    id varchar(64) PRIMARY KEY,
    meta jsonb NOT NULL,
    state jsonb,
    revision bigint NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS meeting_records_updated_idx ON meeting_records (updated_at DESC);

CREATE TABLE IF NOT EXISTS meeting_segments (
    meeting_id varchar(64) NOT NULL REFERENCES meeting_records(id) ON DELETE CASCADE,
    segment_id bigint NOT NULL,
    start_sample bigint NOT NULL,
    end_sample bigint NOT NULL,
    speaker_id integer,
    content text NOT NULL DEFAULT '',
    PRIMARY KEY (meeting_id, segment_id)
);
CREATE INDEX IF NOT EXISTS meeting_segments_time_idx ON meeting_segments (meeting_id, start_sample);

CREATE TABLE IF NOT EXISTS library_records (
    kind varchar(16) PRIMARY KEY,
    content jsonb NOT NULL,
    revision bigint NOT NULL DEFAULT 0
);
