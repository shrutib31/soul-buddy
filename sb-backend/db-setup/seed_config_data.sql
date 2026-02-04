BEGIN;

-- =====================================================
-- 1. SEVERITY LEVELS
-- =====================================================

INSERT INTO severity_levels (severity, description)
VALUES
  ('low',    'Mild, short-lived distress'),
  ('medium', 'Sustained distress impacting functioning'),
  ('high',   'Severe distress or elevated risk')
ON CONFLICT (severity)
DO UPDATE SET description = EXCLUDED.description;


-- =====================================================
-- 2. INTENTS
-- =====================================================

INSERT INTO intents (intent_id, description)
VALUES
  ('VENT',               'User wants to express and be heard'),
  ('SEEK_UNDERSTANDING', 'User wants explanation or clarity'),
  ('OPEN_TO_SOLUTION',   'User is receptive to suggestions'),
  ('TRY_TOOL',           'User explicitly wants to try a tool'),
  ('SEEK_SUPPORT',       'User wants external or human help'),
  ('UNCLEAR',            'Intent is not yet clear')
ON CONFLICT (intent_id)
DO UPDATE SET description = EXCLUDED.description;


-- =====================================================
-- 3. CONVERSATION PHASES
-- =====================================================

INSERT INTO conversation_phases (phase_id, description)
VALUES
  ('CONTAINMENT',        'Emotional safety and exploration'),
  ('CLARIFICATION',      'Pattern and cycle recognition'),
  ('PERSPECTIVE_SHIFT',  'Meaning-making and reframing'),
  ('PSYCHOEDUCATION',    'Understanding mechanisms'),
  ('ACTION',             'Tools, exercises, next steps'),
  ('CRISIS',             'Immediate safety and support')
ON CONFLICT (phase_id)
DO UPDATE SET description = EXCLUDED.description;


-- =====================================================
-- 4. CONVERSATION STEPS
-- =====================================================

INSERT INTO conversation_steps (step_id, phase_id, description)
VALUES
  ('EXPLORATION',        'CONTAINMENT',       'Open-ended exploration'),
  ('EMOTIONS',           'CONTAINMENT',       'Emotion identification'),
  ('BODY',               'CONTAINMENT',       'Somatic awareness'),
  ('THOUGHTS',           'CLARIFICATION',     'Thought identification'),
  ('BEHAVIORS',          'CLARIFICATION',     'Behavior mapping'),
  ('GENTLE_SUMMARY',     'CLARIFICATION',     'Pattern summarization'),
  ('PERSPECTIVE_SHIFT',  'PERSPECTIVE_SHIFT', 'Alternative perspective'),
  ('PSYCHOEDUCATION',    'PSYCHOEDUCATION',   'Explain why this happens'),
  ('REDIRECT_TO_TOOL',   'ACTION',            'Offer an exercise or tool'),
  ('ACKNOWLEDGE_RISK',   'CRISIS',            'Acknowledge risk explicitly'),
  ('ENCOURAGE_SUPPORT',  'CRISIS',            'Encourage external support')
ON CONFLICT (step_id)
DO UPDATE SET
  phase_id    = EXCLUDED.phase_id,
  description = EXCLUDED.description;


-- =====================================================
-- 5. SITUATIONS
-- =====================================================

INSERT INTO situations (situation_id, category, description, is_crisis)
VALUES
  ('FIRST_YEAR_OVERWHELM',   'academic', 'Transition stress in first year', FALSE),
  ('ACADEMIC_COMPARISON',    'academic', 'Feeling behind peers academically', FALSE),
  ('EXAM_ANXIETY',           'academic', 'Exam-related anxiety', FALSE),
  ('GENERAL_OVERWHELM',      'emotional','Diffuse overwhelm across domains', FALSE),
  ('LOW_MOTIVATION',         'emotional','Exhaustion and reduced drive', FALSE),
  ('BELONGING_DOUBT',        'social',   'Feeling out of place or not fitting in', FALSE),
  ('UNLABELED_DISTRESS',     'fallback', 'Vague or unclear distress', FALSE),
  ('PASSIVE_DEATH_WISH',     'crisis',   'Passive death-related thoughts', TRUE)
ON CONFLICT (situation_id)
DO UPDATE SET
  category    = EXCLUDED.category,
  description = EXCLUDED.description,
  is_crisis   = EXCLUDED.is_crisis;


-- =====================================================
-- 6. FLOWS
-- =====================================================

INSERT INTO flows (
  flow_id,
  flow_mode,
  description,
  is_crisis,
  allow_tools,
  allow_psychoeducation
)
VALUES
  ('FLOW_FIRST_YEAR_OVERWHELM', 'containment_first', 'Guided first-year adjustment', FALSE, TRUE,  TRUE),
  ('FLOW_GENERAL_OVERWHELM',    'containment_first', 'General overwhelm support', FALSE, TRUE,  TRUE),
  ('FLOW_EMOTIONAL_OFFLOAD',    'containment_only',  'Pure emotional offload flow', FALSE, FALSE, FALSE),
  ('FLOW_UNLABELED_OVERWHELM',  'containment_only',  'Fallback safe containment flow', FALSE, FALSE, FALSE),
  ('FLOW_CRISIS_HIGH',          'crisis',            'High-risk crisis response', TRUE,  FALSE, FALSE)
ON CONFLICT (flow_id)
DO UPDATE SET
  flow_mode              = EXCLUDED.flow_mode,
  description            = EXCLUDED.description,
  is_crisis              = EXCLUDED.is_crisis,
  allow_tools            = EXCLUDED.allow_tools,
  allow_psychoeducation = EXCLUDED.allow_psychoeducation;


-- =====================================================
-- 7. FLOW STEPS (FSM ORDER)
-- =====================================================

INSERT INTO flow_steps (flow_id, step_order, step_id)
VALUES
  -- First year overwhelm
  ('FLOW_FIRST_YEAR_OVERWHELM', 1, 'EXPLORATION'),
  ('FLOW_FIRST_YEAR_OVERWHELM', 2, 'EMOTIONS'),
  ('FLOW_FIRST_YEAR_OVERWHELM', 3, 'BODY'),
  ('FLOW_FIRST_YEAR_OVERWHELM', 4, 'THOUGHTS'),
  ('FLOW_FIRST_YEAR_OVERWHELM', 5, 'BEHAVIORS'),
  ('FLOW_FIRST_YEAR_OVERWHELM', 6, 'GENTLE_SUMMARY'),
  ('FLOW_FIRST_YEAR_OVERWHELM', 7, 'PERSPECTIVE_SHIFT'),
  ('FLOW_FIRST_YEAR_OVERWHELM', 8, 'PSYCHOEDUCATION'),
  ('FLOW_FIRST_YEAR_OVERWHELM', 9, 'REDIRECT_TO_TOOL'),

  -- Unlabeled overwhelm
  ('FLOW_UNLABELED_OVERWHELM',  1, 'EXPLORATION'),
  ('FLOW_UNLABELED_OVERWHELM',  2, 'EMOTIONS'),
  ('FLOW_UNLABELED_OVERWHELM',  3, 'BODY'),
  ('FLOW_UNLABELED_OVERWHELM',  4, 'GENTLE_SUMMARY'),

  -- Crisis
  ('FLOW_CRISIS_HIGH',          1, 'ACKNOWLEDGE_RISK'),
  ('FLOW_CRISIS_HIGH',          2, 'ENCOURAGE_SUPPORT')
ON CONFLICT (flow_id, step_order)
DO NOTHING;


-- =====================================================
-- 8. FLOW–SITUATION MAPPING (POLICY)
-- =====================================================

INSERT INTO flow_situation_mapping (
  situation_id,
  severity,
  flow_id,
  confidence_min
)
VALUES
  ('FIRST_YEAR_OVERWHELM', 'low',    'FLOW_EMOTIONAL_OFFLOAD',    0.60),
  ('FIRST_YEAR_OVERWHELM', 'medium', 'FLOW_FIRST_YEAR_OVERWHELM', 0.65),
  ('FIRST_YEAR_OVERWHELM', 'high',   'FLOW_GENERAL_OVERWHELM',    0.65),

  ('GENERAL_OVERWHELM',    'low',    'FLOW_EMOTIONAL_OFFLOAD',    0.60),
  ('GENERAL_OVERWHELM',    'medium', 'FLOW_GENERAL_OVERWHELM',    0.65),
  ('GENERAL_OVERWHELM',    'high',   'FLOW_CRISIS_HIGH',          0.70),

  ('UNLABELED_DISTRESS',   'low',    'FLOW_UNLABELED_OVERWHELM',  0.00),
  ('UNLABELED_DISTRESS',   'medium', 'FLOW_UNLABELED_OVERWHELM',  0.00),
  ('UNLABELED_DISTRESS',   'high',   'FLOW_UNLABELED_OVERWHELM',  0.00),

  ('PASSIVE_DEATH_WISH',   'high',   'FLOW_CRISIS_HIGH',          0.00)
ON CONFLICT (situation_id, severity, flow_id)
DO UPDATE SET
  confidence_min = EXCLUDED.confidence_min;

COMMIT;
