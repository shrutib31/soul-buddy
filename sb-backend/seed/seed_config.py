import logging
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from orm.models import (
    Situation,
    SeverityLevel,
    Flow,
    FlowStep,
    FlowSituationMapping,
    ConversationPhase,
    ConversationStep,
    Intent,
)

logger = logging.getLogger(__name__)


# -------------------------
# 1. Severity Levels
# -------------------------
def seed_severity_levels(db: Session):
    logger.info("[1/8] Seeding severity levels...")
    data = [
        {"severity": "low", "description": "Mild, short-lived distress"},
        {"severity": "medium", "description": "Sustained distress impacting functioning"},
        {"severity": "high", "description": "Severe distress or elevated risk"},
    ]

    stmt = insert(SeverityLevel).values(data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["severity"],
        set_={"description": stmt.excluded.description},
    )
    db.execute(stmt)
    db.commit()
    logger.info(f"✅ Seeded {len(data)} severity levels")



# -------------------------
# 2. Intents
# -------------------------
def seed_intents(db: Session):
    logger.info("[2/8] Seeding intents...")
    data = [
        {"intent_id": "GREETING", "description": "User is greeting or saying hello"},
        {"intent_id": "VENTING", "description": "User wants to express and be heard"},
        {"intent_id": "SEEK_INFORMATION", "description": "User is asking for information or clarification"},
        {"intent_id": "SEEK_UNDERSTANDING", "description": "User wants explanation or clarity"},
        {"intent_id": "OPEN_TO_SOLUTION", "description": "User is receptive to suggestions"},
        {"intent_id": "TRY_TOOL", "description": "User explicitly wants to try a tool"},
        {"intent_id": "SEEK_SUPPORT", "description": "User wants external or human help"},
        {"intent_id": "UNCLEAR", "description": "Intent is not yet clear"},
    ]

    stmt = insert(Intent).values(data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["intent_id"],
        set_={"description": stmt.excluded.description},
    )
    db.execute(stmt)
    db.commit()
    logger.info(f"✅ Seeded {len(data)} intents")



# -------------------------
# 3. Conversation Phases
# -------------------------
def seed_conversation_phases(db: Session):
    logger.info("[3/8] Seeding conversation phases...")
    data = [
        {"phase_id": "CONTAINMENT", "description": "Emotional safety and exploration"},
        {"phase_id": "CLARIFICATION", "description": "Pattern and cycle recognition"},
        {"phase_id": "PERSPECTIVE_SHIFT", "description": "Meaning-making and reframing"},
        {"phase_id": "PSYCHOEDUCATION", "description": "Understanding mechanisms"},
        {"phase_id": "ACTION", "description": "Tools, exercises, next steps"},
        {"phase_id": "CRISIS", "description": "Immediate safety and support"},
    ]

    stmt = insert(ConversationPhase).values(data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["phase_id"],
        set_={"description": stmt.excluded.description},
    )
    db.execute(stmt)
    db.commit()
    logger.info(f"✅ Seeded {len(data)} conversation phases")



# -------------------------
# 4. Conversation Steps
# -------------------------
def seed_conversation_steps(db: Session):
    logger.info("[4/8] Seeding conversation steps...")
    data = [
        {"step_id": "EXPLORATION", "phase_id": "CONTAINMENT", "description": "Open-ended exploration"},
        {"step_id": "EMOTIONS", "phase_id": "CONTAINMENT", "description": "Emotion identification"},
        {"step_id": "BODY", "phase_id": "CONTAINMENT", "description": "Somatic awareness"},
        {"step_id": "THOUGHTS", "phase_id": "CLARIFICATION", "description": "Thought identification"},
        {"step_id": "BEHAVIORS", "phase_id": "CLARIFICATION", "description": "Behavior mapping"},
        {"step_id": "GENTLE_SUMMARY", "phase_id": "CLARIFICATION", "description": "Pattern summarization"},
        {"step_id": "PERSPECTIVE_SHIFT", "phase_id": "PERSPECTIVE_SHIFT", "description": "Alternative perspective"},
        {"step_id": "PSYCHOEDUCATION", "phase_id": "PSYCHOEDUCATION", "description": "Explain why this happens"},
        {"step_id": "REDIRECT_TO_TOOL", "phase_id": "ACTION", "description": "Offer an exercise or tool"},
        {"step_id": "ACKNOWLEDGE_RISK", "phase_id": "CRISIS", "description": "Acknowledge risk explicitly"},
        {"step_id": "ENCOURAGE_SUPPORT", "phase_id": "CRISIS", "description": "Encourage external support"},
    ]

    stmt = insert(ConversationStep).values(data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["step_id"],
        set_={
            "phase_id": stmt.excluded.phase_id,
            "description": stmt.excluded.description,
        },
    )
    db.execute(stmt)
    db.commit()
    logger.info(f"✅ Seeded {len(data)} conversation steps")



# -------------------------
# 5. Situations
# -------------------------
def seed_situations(db: Session):
    logger.info("[5/8] Seeding situations...")
    data = [
        {"situation_id": "FIRST_YEAR_OVERWHELM", "category": "academic", "description": "Transition stress in first year", "is_crisis": False},
        {"situation_id": "ACADEMIC_COMPARISON", "category": "academic", "description": "Feeling behind peers academically", "is_crisis": False},
        {"situation_id": "EXAM_ANXIETY", "category": "academic", "description": "Exam-related anxiety", "is_crisis": False},
        {"situation_id": "GENERAL_OVERWHELM", "category": "emotional", "description": "Diffuse overwhelm across domains", "is_crisis": False},
        {"situation_id": "LOW_MOTIVATION", "category": "emotional", "description": "Exhaustion and reduced drive", "is_crisis": False},
        {"situation_id": "BELONGING_DOUBT", "category": "social", "description": "Feeling out of place or not fitting in", "is_crisis": False},
        {"situation_id": "UNLABELED_DISTRESS", "category": "fallback", "description": "Vague or unclear distress", "is_crisis": False},
        {"situation_id": "PASSIVE_DEATH_WISH", "category": "crisis", "description": "Passive death-related thoughts", "is_crisis": True},
    ]

    stmt = insert(Situation).values(data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["situation_id"],
        set_={
            "category": stmt.excluded.category,
            "description": stmt.excluded.description,
            "is_crisis": stmt.excluded.is_crisis,
        },
    )
    db.execute(stmt)
    db.commit()
    logger.info(f"✅ Seeded {len(data)} situations")



# -------------------------
# 6. Flows
# -------------------------
def seed_flows(db: Session):
    logger.info("[6/8] Seeding flows...")
    data = [
        {"flow_id": "FLOW_FIRST_YEAR_OVERWHELM", "flow_mode": "containment_first", "description": "Guided first-year adjustment", "is_crisis": False, "allow_tools": True, "allow_psychoeducation": True},
        {"flow_id": "FLOW_GENERAL_OVERWHELM", "flow_mode": "containment_first", "description": "General overwhelm support", "is_crisis": False, "allow_tools": True, "allow_psychoeducation": True},
        {"flow_id": "FLOW_EMOTIONAL_OFFLOAD", "flow_mode": "containment_only", "description": "Pure emotional offload flow", "is_crisis": False, "allow_tools": False, "allow_psychoeducation": False},
        {"flow_id": "FLOW_UNLABELED_OVERWHELM", "flow_mode": "containment_only", "description": "Fallback safe containment flow", "is_crisis": False, "allow_tools": False, "allow_psychoeducation": False},
        {"flow_id": "FLOW_CRISIS_HIGH", "flow_mode": "crisis", "description": "High-risk crisis response", "is_crisis": True, "allow_tools": False, "allow_psychoeducation": False},
    ]

    stmt = insert(Flow).values(data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["flow_id"],
        set_={
            "flow_mode": stmt.excluded.flow_mode,
            "description": stmt.excluded.description,
            "is_crisis": stmt.excluded.is_crisis,
            "allow_tools": stmt.excluded.allow_tools,
            "allow_psychoeducation": stmt.excluded.allow_psychoeducation,
        },
    )
    db.execute(stmt)
    db.commit()
    logger.info(f"✅ Seeded {len(data)} flows")



# -------------------------
# 7. Flow Steps (FSM order)
# -------------------------
def seed_flow_steps(db: Session):
    logger.info("[7/8] Seeding flow steps...")
    data = [
        {"flow_id": "FLOW_FIRST_YEAR_OVERWHELM", "step_order": 1, "step_id": "EXPLORATION"},
        {"flow_id": "FLOW_FIRST_YEAR_OVERWHELM", "step_order": 2, "step_id": "EMOTIONS"},
        {"flow_id": "FLOW_FIRST_YEAR_OVERWHELM", "step_order": 3, "step_id": "BODY"},
        {"flow_id": "FLOW_FIRST_YEAR_OVERWHELM", "step_order": 4, "step_id": "THOUGHTS"},
        {"flow_id": "FLOW_FIRST_YEAR_OVERWHELM", "step_order": 5, "step_id": "BEHAVIORS"},
        {"flow_id": "FLOW_FIRST_YEAR_OVERWHELM", "step_order": 6, "step_id": "GENTLE_SUMMARY"},
        {"flow_id": "FLOW_FIRST_YEAR_OVERWHELM", "step_order": 7, "step_id": "PERSPECTIVE_SHIFT"},
        {"flow_id": "FLOW_FIRST_YEAR_OVERWHELM", "step_order": 8, "step_id": "PSYCHOEDUCATION"},
        {"flow_id": "FLOW_FIRST_YEAR_OVERWHELM", "step_order": 9, "step_id": "REDIRECT_TO_TOOL"},
        {"flow_id": "FLOW_UNLABELED_OVERWHELM", "step_order": 1, "step_id": "EXPLORATION"},
        {"flow_id": "FLOW_UNLABELED_OVERWHELM", "step_order": 2, "step_id": "EMOTIONS"},
        {"flow_id": "FLOW_UNLABELED_OVERWHELM", "step_order": 3, "step_id": "BODY"},
        {"flow_id": "FLOW_UNLABELED_OVERWHELM", "step_order": 4, "step_id": "GENTLE_SUMMARY"},
        {"flow_id": "FLOW_CRISIS_HIGH", "step_order": 1, "step_id": "ACKNOWLEDGE_RISK"},
        {"flow_id": "FLOW_CRISIS_HIGH", "step_order": 2, "step_id": "ENCOURAGE_SUPPORT"},
    ]

    stmt = insert(FlowStep).values(data)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=["flow_id", "step_order"]
    )
    db.execute(stmt)
    db.commit()
    logger.info(f"✅ Seeded {len(data)} flow steps")



# -------------------------
# 8. Flow–Situation Mapping (Policy)
# -------------------------
def seed_flow_situation_mapping(db: Session):
    logger.info("[8/8] Seeding flow–situation mappings...")
    data = [
        {"situation_id": "FIRST_YEAR_OVERWHELM", "severity": "low", "flow_id": "FLOW_EMOTIONAL_OFFLOAD", "confidence_min": 0.60},
        {"situation_id": "FIRST_YEAR_OVERWHELM", "severity": "medium", "flow_id": "FLOW_FIRST_YEAR_OVERWHELM", "confidence_min": 0.65},
        {"situation_id": "FIRST_YEAR_OVERWHELM", "severity": "high", "flow_id": "FLOW_GENERAL_OVERWHELM", "confidence_min": 0.65},
        {"situation_id": "GENERAL_OVERWHELM", "severity": "low", "flow_id": "FLOW_EMOTIONAL_OFFLOAD", "confidence_min": 0.60},
        {"situation_id": "GENERAL_OVERWHELM", "severity": "medium", "flow_id": "FLOW_GENERAL_OVERWHELM", "confidence_min": 0.65},
        {"situation_id": "GENERAL_OVERWHELM", "severity": "high", "flow_id": "FLOW_CRISIS_HIGH", "confidence_min": 0.70},
        {"situation_id": "UNLABELED_DISTRESS", "severity": "low", "flow_id": "FLOW_UNLABELED_OVERWHELM", "confidence_min": 0.0},
        {"situation_id": "UNLABELED_DISTRESS", "severity": "medium", "flow_id": "FLOW_UNLABELED_OVERWHELM", "confidence_min": 0.0},
        {"situation_id": "UNLABELED_DISTRESS", "severity": "high", "flow_id": "FLOW_UNLABELED_OVERWHELM", "confidence_min": 0.0},
        {"situation_id": "PASSIVE_DEATH_WISH", "severity": "high", "flow_id": "FLOW_CRISIS_HIGH", "confidence_min": 0.0},
    ]

    stmt = insert(FlowSituationMapping).values(data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["situation_id", "severity", "flow_id"],
        set_={"confidence_min": stmt.excluded.confidence_min},
    )
    db.execute(stmt)
    db.commit()
    logger.info(f"✅ Seeded {len(data)} flow–situation mappings")



# -------------------------
# Master seed function
# -------------------------
def seed_all_config(db: Session):
    logger.info("="*80)
    logger.info("🌱 STARTING CONFIGURATION DATA SEEDING")
    logger.info("="*80)
    
    try:
        seed_severity_levels(db)
        seed_intents(db)
        seed_conversation_phases(db)
        seed_conversation_steps(db)
        seed_situations(db)
        seed_flows(db)
        seed_flow_steps(db)
        seed_flow_situation_mapping(db)
        
        logger.info("="*80)
        logger.info("✅ ALL CONFIGURATION DATA SEEDED SUCCESSFULLY")
        logger.info("="*80)
    except Exception as error:
        logger.error(f"❌ Configuration seeding failed: {error}")
        raise
