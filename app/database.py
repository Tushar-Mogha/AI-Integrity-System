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
    # table is already created in Supabase dashboard
    # this function kept for compatibility
    pass

def save_result(result):
    try:
        client = get_client()
        record = {
            "student_id"    : result["student_id"],
            "student_name"  : result["student_name"],
            "module1_score" : result["module1_ai_score"],
            "module2_score" : result["module2_style_score"],
            "module2_label" : result.get("module2_label", ""),
            "module3_score" : result["module3_behavior_score"],
            "composite_score": result["composite_score"],
            "risk_level"    : result["risk_level"],
            "behavior_label": result["behavior_label"],
            "grade_jump"    : result["grade_jump"],
            "g1"            : result["G1"],
            "g2"            : result["G2"],
            "g3"            : result["G3"],
            "absences"      : result["absences"],
            "studytime"     : result["studytime"],
            "failures"      : result["failures"],
            "essay_text"    : result.get("essay_text", ""),
            "faculty_note"  : "",
            "analyzed_at"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        client.table("risk_results").insert(record).execute()
        return True
    except Exception as e:
        print(f"Database save error: {e}")
        return False

def update_note(record_id, note):
    try:
        client = get_client()
        client.table("risk_results").update(
            {"faculty_note": note}
        ).eq("id", record_id).execute()
        return True
    except Exception as e:
        print(f"Database update error: {e}")
        return False

def get_all_results():
    try:
        client   = get_client()
        response = client.table("risk_results").select(
            "id, student_id, student_name, module1_score, module2_score, "
            "module2_label, module3_score, composite_score, risk_level, "
            "behavior_label, grade_jump, g1, g2, g3, absences, studytime, "
            "failures, faculty_note, analyzed_at"
        ).order("analyzed_at", desc=True).execute()

        # normalize keys to uppercase G1 G2 G3 for dashboard compatibility
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
            "student_id", student_id
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
        client = get_client()
        client.table("risk_results").delete().eq("id", record_id).execute()
        return True
    except Exception as e:
        print(f"Database delete error: {e}")
        return False