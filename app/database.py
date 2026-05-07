# Database helper for AcademicGuard
# Uses Supabase PostgreSQL for persistent cloud storage
# Team - Abhinandan, Stuti, Tushar

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def get_client():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def init_db():
    pass

def save_result(result):
    try:
        client = get_client()
        # use UTC time for consistency
        record = {
            "student_id"    : str(result["student_id"]),
            "student_name"  : str(result["student_name"]),
            "module1_score" : float(result["module1_ai_score"]),
            "module2_score" : float(result["module2_style_score"]),
            "module2_label" : str(result.get("module2_label", "")),
            "module3_score" : float(result["module3_behavior_score"]),
            "composite_score": float(result["composite_score"]),
            "risk_level"    : str(result["risk_level"]),
            "behavior_label": str(result["behavior_label"]),
            "module3_reason": result.get("module3_reason", ""),
            "grade_jump"    : float(result["grade_jump"]),
            "g1"            : int(result["G1"]),
            "g2"            : int(result["G2"]),
            "g3"            : int(result["G3"]),
            "absences"      : int(result["absences"]),
            "studytime"     : int(result["studytime"]),
            "failures"      : int(result["failures"]),
            "essay_text"    : str(result.get("essay_text", "")),
            "faculty_note"  : "",
            "analyzed_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        response = client.table("risk_results").insert(record).execute()
        if response.data:
            print(f"Saved student: {result['student_name']} with ID: {response.data[0]['id']}")
            return response.data[0]['id']  # return the new record id
        return None
    except Exception as e:
        print(f"Database save error: {e}")
        return None

def update_note(record_id, note):
    try:
        print(f"Updating record ID: {record_id} with note: {note}")  # 👈 ADD

        client = get_client()
        response = client.table("risk_results").update(
            {"faculty_note": str(note)}
        ).eq("id", int(record_id)).execute()

        print(f"Update response: {response.data}")  # 👈 ADD

        return len(response.data) > 0
    except Exception as e:
        print(f"Database update error: {e}")
        return False

def get_all_results():
    try:
        client   = get_client()
        response = client.table("risk_results").select("*").order(
            "analyzed_at", desc=True
        ).execute()
        records = []
        for r in response.data:
            r["G1"] = r.pop("g1", 0)
            r["G2"] = r.pop("g2", 0)
            r["G3"] = r.pop("g3", 0)
            records.append(r)
        return records
    except Exception as e:
        print(f"Database fetch error: {e}")
        return []

def get_student_history(student_id):
    try:
        client   = get_client()
        response = client.table("risk_results").select("*").eq(
            "student_id", str(student_id)
        ).order("analyzed_at", desc=True).execute()
        records = []
        for r in response.data:
            r["G1"] = r.pop("g1", 0)
            r["G2"] = r.pop("g2", 0)
            r["G3"] = r.pop("g3", 0)
            records.append(r)
        return records
    except Exception as e:
        print(f"Database fetch error: {e}")
        return []

def delete_record(record_id):
    try:
        client   = get_client()
        response = client.table("risk_results").delete().eq(
            "id", int(record_id)
        ).execute()
        return True
    except Exception as e:
        print(f"Database delete error: {e}")
        return False