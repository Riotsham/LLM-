import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

try:
    from pymongo import MongoClient
except ModuleNotFoundError:
    MongoClient = None
try:
    from bson import ObjectId
except Exception:
    ObjectId = None

MONGO_URI = "mongodb://localhost:27017"
_STORE_PATH = Path(__file__).resolve().parent / "assistant_db_local.json"


if MongoClient is not None:
    client = MongoClient(MONGO_URI)
    db = client["assistant_db"]
    users_collection = db["users"]
    sessions_collection = db["sessions"]
    pitch_collection = db["pitch_logs"]
else:
    client = None
    db = None
    users_collection = None
    sessions_collection = None
    pitch_collection = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_store() -> dict:
    if not _STORE_PATH.exists():
        return {"users": [], "sessions": [], "pitch_logs": []}
    try:
        data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        data.setdefault("users", [])
        data.setdefault("sessions", [])
        data.setdefault("pitch_logs", [])
        return data
    except Exception:
        return {"users": [], "sessions": [], "pitch_logs": []}


def _save_store(data: dict) -> None:
    _STORE_PATH.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def _find_user_index(users: list[dict], name: str) -> int:
    target = (name or "").strip().lower()
    for i, user in enumerate(users):
        if str(user.get("name", "")).strip().lower() == target:
            return i
    return -1


def _normalize_id(value):
    if value is None:
        return None
    return str(value)


def _to_object_id(value):
    if ObjectId is None:
        return value
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return value


def get_user_by_name(name: str) -> dict | None:
    if not name:
        return None
    if users_collection is not None:
        return users_collection.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}})

    store = _load_store()
    idx = _find_user_index(store["users"], name)
    return dict(store["users"][idx]) if idx >= 0 else None


def get_existing_user() -> dict | None:
    if users_collection is not None:
        return users_collection.find_one(sort=[("created_at", 1)])

    store = _load_store()
    users = store.get("users", [])
    if not users:
        return None
    return dict(users[0])


def create_user(profile: dict) -> dict:
    doc = dict(profile or {})
    doc.setdefault("onboarding_complete", False)
    doc.setdefault("created_at", _now_iso())

    if users_collection is not None:
        result = users_collection.insert_one(doc)
        doc["_id"] = result.inserted_id
        return doc

    store = _load_store()
    doc["_id"] = doc.get("_id") or str(uuid4())
    store["users"].append(doc)
    _save_store(store)
    return doc


def update_user(name: str, updated_fields: dict):
    if not name or not updated_fields:
        return
    if users_collection is not None:
        users_collection.update_one({"name": {"$regex": f"^{name}$", "$options": "i"}}, {"$set": updated_fields})
        return

    store = _load_store()
    idx = _find_user_index(store["users"], name)
    if idx < 0:
        return
    store["users"][idx].update(updated_fields)
    _save_store(store)


def update_user_by_id(user_id, updated_fields: dict):
    if not user_id or not updated_fields:
        return
    if users_collection is not None:
        users_collection.update_one({"_id": _to_object_id(user_id)}, {"$set": updated_fields})
        return

    store = _load_store()
    target = _normalize_id(user_id)
    for user in store["users"]:
        if _normalize_id(user.get("_id")) == target:
            user.update(updated_fields)
            _save_store(store)
            return


def log_pitch(user_id, avg_pitch: float):
    doc = {
        "_id": str(uuid4()),
        "user_id": user_id,
        "timestamp": _now_iso(),
        "pitch": float(avg_pitch or 0.0),
    }

    if pitch_collection is not None:
        mongo_doc = dict(doc)
        mongo_doc["user_id"] = _to_object_id(user_id)
        mongo_doc.pop("_id", None)
        pitch_collection.insert_one(mongo_doc)
        return

    store = _load_store()
    store["pitch_logs"].append(doc)
    _save_store(store)


def save_full_session(user_id, conversation: list, pitch_values: list[float], metadata: dict | None = None):
    clean_values = [float(v) for v in (pitch_values or []) if isinstance(v, (int, float))]
    average_pitch_session = sum(clean_values) / len(clean_values) if clean_values else 0.0

    doc = {
        "_id": str(uuid4()),
        "user_id": user_id,
        "timestamp": _now_iso(),
        "conversation": conversation or [],
        "average_pitch_session": float(average_pitch_session),
    }
    if metadata:
        doc["metadata"] = dict(metadata)

    if sessions_collection is not None:
        mongo_doc = dict(doc)
        mongo_doc["user_id"] = _to_object_id(user_id)
        mongo_doc.pop("_id", None)
        sessions_collection.insert_one(mongo_doc)
        return

    store = _load_store()
    store["sessions"].append(doc)
    _save_store(store)


def log_session(user_id, transcript: str, avg_pitch: float):
    # Backward-compatible logger used by older call sites.
    conversation = [("session_text", transcript or "")]
    save_full_session(user_id=user_id, conversation=conversation, pitch_values=[float(avg_pitch or 0.0)])


def update_pitch_baseline(user_id, baseline: float):
    if users_collection is not None:
        users_collection.update_one(
            {"_id": _to_object_id(user_id)},
            {"$set": {"pitch_baseline": float(baseline or 0.0)}},
        )
        return

    store = _load_store()
    target = _normalize_id(user_id)
    for user in store["users"]:
        if _normalize_id(user.get("_id")) == target:
            user["pitch_baseline"] = float(baseline or 0.0)
            _save_store(store)
            return


def get_all_users() -> list[dict]:
    if users_collection is not None:
        users = list(users_collection.find({}))
        users.sort(key=lambda u: str(u.get("created_at", "")))
        return users

    store = _load_store()
    users = [dict(u) for u in store.get("users", [])]
    users.sort(key=lambda u: str(u.get("created_at", "")))
    return users


def get_sessions_by_user_id(user_id, limit: int | None = None) -> list[dict]:
    if not user_id:
        return []

    if sessions_collection is not None:
        cursor = sessions_collection.find({"user_id": _to_object_id(user_id)}).sort("timestamp", 1)
        if isinstance(limit, int) and limit > 0:
            docs = list(cursor.limit(limit))
        else:
            docs = list(cursor)
        return docs

    store = _load_store()
    target = _normalize_id(user_id)
    docs = [dict(s) for s in store.get("sessions", []) if _normalize_id(s.get("user_id")) == target]
    docs.sort(key=lambda s: str(s.get("timestamp", "")))
    if isinstance(limit, int) and limit > 0:
        return docs[:limit]
    return docs
