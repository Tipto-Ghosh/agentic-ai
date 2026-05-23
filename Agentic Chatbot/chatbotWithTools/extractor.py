import json
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from memory import save_fact

load_dotenv()

_extractor = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

_PROMPT_TEMPLATE = """\
From this conversation exchange, extract any personal facts about the user \
that are worth remembering long-term (name, job, preferences, goals, location, etc.).

Return ONLY a valid JSON list of strings — no markdown, no explanation.
Return [] if nothing important was shared.

User: {human_msg}
Assistant: {ai_reply}\
"""


def extract_and_save_facts(user_id: str, human_msg: str, ai_reply: str) -> None:
    """
    Parse a conversation turn with the LLM, extract memorable user facts,
    and persist each one to MongoDB via save_fact().
    """
    prompt = _PROMPT_TEMPLATE.format(human_msg=human_msg, ai_reply=ai_reply)

    try:
        raw = _extractor.invoke(prompt).content
        facts = json.loads(raw)
        if isinstance(facts, list):
            for fact in facts:
                if isinstance(fact, str) and fact.strip():
                    save_fact(user_id, fact.strip())
    except (json.JSONDecodeError, Exception):
        pass
