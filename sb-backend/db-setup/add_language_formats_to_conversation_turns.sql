/*
-- Migration to add multi-format content columns to conversation_turns
ALTER TABLE conversation_turns 
ADD COLUMN IF NOT EXISTS romanised_content TEXT,
ADD COLUMN IF NOT EXISTS canonical_content TEXT,
ADD COLUMN IF NOT EXISTS mixed_content TEXT;
*/
