-- Migration to add language column to conversation_turns table
ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS language VARCHAR(10) DEFAULT 'en-IN';

-- Initial comment to explain usage
COMMENT ON COLUMN conversation_turns.language IS 'Language code of this specific message (e.g., en-IN, hi-IN, bn-IN)';
