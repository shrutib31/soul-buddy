-- DROP TABLES SCRIPT --
-- Drop tables in reverse order of dependencies to avoid foreign key constraint errors

-- Runtime tables
DROP TABLE IF EXISTS step_outputs;
DROP TABLE IF EXISTS crisis_events;
DROP TABLE IF EXISTS flow_state;
DROP TABLE IF EXISTS risk_assessments;
DROP TABLE IF EXISTS situation_assessments;
DROP TABLE IF EXISTS conversation_turns;

--Audit log
DROP TABLE IF EXISTS encryption_audit_log;

-- Core conversation table
DROP TABLE IF EXISTS sb_conversations;

-- Configuration tables
DROP TABLE IF EXISTS flow_situation_mapping;
DROP TABLE IF EXISTS flow_steps;
DROP TABLE IF EXISTS flows;
DROP TABLE IF EXISTS conversation_steps;
DROP TABLE IF EXISTS conversation_phases;
DROP TABLE IF EXISTS intents;
DROP TABLE IF EXISTS severity_levels;
DROP TABLE IF EXISTS situations;