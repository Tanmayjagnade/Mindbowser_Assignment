"""
Agentic workflow router with guardrails.

Flow:
  1. Greeting?          -> friendly welcome (no LLM / RAG needed)
  2. Out-of-scope?      -> guardrail blocks it immediately
  3. Appointment?       -> check_available_slots() mock tool
  4. Healthcare query   -> RAG pipeline + LLM
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Greeting detection
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|howdy|hiya|greetings|good\s*(morning|afternoon|evening|day))\s*[!.,?]*\s*$",
    r"^\s*what'?s\s+up\s*[!.,?]*\s*$",
    r"^\s*namaste\s*[!.,?]*\s*$",
]

_GREETING_REPLY = (
    "Hello! I'm MedAssist, your healthcare AI assistant. 👋\n\n"
    "I can help you with:\n"
    "• Questions about healthcare policies (HIPAA, discharge, insurance)\n"
    "• Medication refill procedures\n"
    "• Telehealth consultation guidelines\n"
    "• Booking appointments\n\n"
    "Please ask me a healthcare-related question to get started!"
)


def _is_greeting(question: str) -> bool:
    q = question.strip().lower()
    return any(re.match(p, q, re.IGNORECASE) for p in _GREETING_PATTERNS)


# ---------------------------------------------------------------------------
# 2. Guardrails — block clearly out-of-scope questions
# ---------------------------------------------------------------------------

# Topics that are completely outside healthcare
_BLOCKED_TOPICS = [
    r"\b(prime\s*minister|president|minister|politician|election|parliament|congress|senate)\b",
    r"\b(cricket|football|soccer|tennis|ipl|sport|match|team|player)\b",
    r"\b(stock|share\s*market|sensex|nifty|crypto|bitcoin|nse|bse)\b",
    r"\b(capital\s*of|geography|country|city|weather|climate|temperature)\b",
    r"\b(movie|film|actor|actress|celebrity|bollywood|hollywood)\b",
    r"\b(recipe|cook|bake|restaurant|food\s*order)\b",
    r"\b(math|science|history|who\s+invented|when\s+was|how\s+old\s+is)\b",
    r"\b(phone|laptop|computer|software|code|programming|python|java)\b",
]

# If ANY of these healthcare signals are present, don't block
_HEALTHCARE_SIGNALS = [
    r"\b(doctor|patient|medicine|medication|hospital|clinic|health|medical|disease|symptom)\b",
    r"\b(hipaa|insurance|discharge|telehealth|appointment|refill|prescription|diagnosis)\b",
    r"\b(surgery|treatment|therapy|nurse|pharmacy|dosage|allergy|fever|pain)\b",
]

_GUARDRAIL_REPLY = (
    "I'm sorry, I can only answer questions related to healthcare — such as "
    "patient rights, medications, appointments, discharge instructions, insurance, "
    "or telehealth guidelines.\n\n"
    "Please ask a healthcare-related question and I'll be happy to help!"
)


def _is_out_of_scope(question: str) -> bool:
    q = question.lower()
    # If any healthcare signal is present, always allow
    if any(re.search(p, q) for p in _HEALTHCARE_SIGNALS):
        return False
    # If any blocked topic matches, block it
    return any(re.search(p, q) for p in _BLOCKED_TOPICS)


# ---------------------------------------------------------------------------
# 3. Appointment intent
# ---------------------------------------------------------------------------

_APPOINTMENT_PATTERNS = [
    r"\bbook\b", r"\bschedule\b", r"\bappointment\b", r"\bslot\b",
    r"\bavailability\b", r"\bavailable\b", r"\bconsult(ation)?\b",
]

_DEPARTMENT_MAP: Dict[str, List[str]] = {
    "Cardiology":       ["cardiology", "cardiologist", "heart", "cardiac"],
    "Orthopedics":      ["orthopedics", "orthopedic", "bone", "joint"],
    "Neurology":        ["neurology", "neurologist", "brain", "nerve"],
    "Dermatology":      ["dermatology", "dermatologist", "skin", "rash"],
    "Pediatrics":       ["pediatrics", "pediatrician", "child", "children"],
    "Ophthalmology":    ["ophthalmology", "eye", "vision"],
    "Gynecology":       ["gynecology", "gynecologist", "obgyn"],
    "General Medicine": ["general", "gp", "primary care", "physician"],
}

_DAY_OFFSETS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_MOCK_SLOTS: Dict[str, List[str]] = {
    "Cardiology":       ["09:00 AM", "11:30 AM", "03:00 PM"],
    "Orthopedics":      ["10:00 AM", "02:00 PM", "04:30 PM"],
    "Neurology":        ["09:30 AM", "01:00 PM", "03:30 PM"],
    "Dermatology":      ["08:30 AM", "11:00 AM", "02:30 PM"],
    "Pediatrics":       ["09:00 AM", "10:30 AM", "01:30 PM"],
    "Ophthalmology":    ["10:00 AM", "12:00 PM", "03:00 PM"],
    "Gynecology":       ["09:00 AM", "11:00 AM", "02:00 PM"],
    "General Medicine": ["08:00 AM", "09:30 AM", "11:00 AM", "02:00 PM", "04:00 PM"],
}


def is_appointment_query(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _APPOINTMENT_PATTERNS)


def _extract_department(question: str) -> str:
    q = question.lower()
    for dept, keywords in _DEPARTMENT_MAP.items():
        if any(kw in q for kw in keywords):
            return dept
    return "General Medicine"


def _extract_date(question: str) -> str:
    q = question.lower()
    today = date.today()
    for day_name, offset in _DAY_OFFSETS.items():
        if day_name in q:
            days_ahead = (offset - today.weekday()) % 7 or 7
            return (today + timedelta(days=days_ahead)).strftime("%A, %B %d %Y")
    if "tomorrow" in q:
        return (today + timedelta(days=1)).strftime("%A, %B %d %Y")
    return "next available date"


def check_available_slots(department: str = "General Medicine", target_date: str | None = None) -> Dict[str, Any]:
    """Mock appointment-availability tool."""
    slots = _MOCK_SLOTS.get(department, _MOCK_SLOTS["General Medicine"])
    day = target_date or str(date.today())
    logger.info("Tool: check_available_slots(dept=%s, date=%s)", department, day)
    return {
        "department": department,
        "date": day,
        "available_slots": slots,
        "booking_instructions": "Call 1800-MED-ASSIST or visit medassist.example.com/appointments to confirm.",
    }


# ---------------------------------------------------------------------------
# HealthcareAgent
# ---------------------------------------------------------------------------

class HealthcareAgent:
    def __init__(self) -> None:
        from app.rag import RAGService
        from app.llm import LLMService
        self.rag = RAGService()
        self.llm = LLMService()

    def handle(self, question: str) -> Dict[str, Any]:
        # --- Step 1: Greeting ---
        if _is_greeting(question):
            logger.info("Intent: greeting")
            return {
                "answer": _GREETING_REPLY,
                "sources": [],
                "confidence": "high",
                "intent": "greeting",
                "tool_used": "none",
            }

        # --- Step 2: Guardrail — block out-of-scope ---
        if _is_out_of_scope(question):
            logger.info("Guardrail blocked: %.60s", question)
            return {
                "answer": _GUARDRAIL_REPLY,
                "sources": [],
                "confidence": "high",
                "intent": "out_of_scope",
                "tool_used": "guardrail",
            }

        # --- Step 3: Appointment booking ---
        if is_appointment_query(question):
            department = _extract_department(question)
            target_date = _extract_date(question)
            tool_result = check_available_slots(department, target_date)
            slots = ", ".join(tool_result["available_slots"])
            answer = (
                f"I checked appointment availability for {department} on {target_date}. "
                f"Available slots: {slots}. {tool_result['booking_instructions']}"
            )
            return {
                "answer": answer,
                "sources": [],
                "confidence": "high",
                "intent": "appointment_booking",
                "tool_used": "check_available_slots",
                "tool_result": tool_result,
            }

        # --- Step 4: RAG knowledge query ---
        logger.info("Intent: knowledge_query — %.60s", question)
        retrieved = self.rag.retrieve(question)
        if not retrieved:
            return {
                "answer": "I could not find this information in the provided documents.",
                "sources": [],
                "confidence": "none",
                "intent": "knowledge_query",
                "tool_used": "rag_retrieval",
            }

        context = self.rag.build_context(retrieved)
        answer = self.llm.answer_with_context(question, context)
        sources = [{"document": r.document, "chunk": r.chunk[:250] + "..."} for r in retrieved]
        return {
            "answer": answer,
            "sources": sources,
            "confidence": self.rag.estimate_confidence(retrieved),
            "intent": "knowledge_query",
            "tool_used": "rag_retrieval",
        }
