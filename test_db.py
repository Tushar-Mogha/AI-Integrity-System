# test_db.py - run this to check if supabase is working
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print("URL found:", "Yes" if url else "NO - URL IS MISSING")
print("KEY found:", "Yes" if key else "NO - KEY IS MISSING")
print("URL value:", url[:30] if url else "None")

try:
    from supabase import create_client
    client = create_client(url, key)
    print("Client created successfully!")

    # test insert
    response = client.table("risk_results").insert({
    "student_id"    : "TEST001",
    "student_name"  : "Test Student",
    "module1_score" : 90.0,
    "module2_score" : 85.0,
    "module2_label" : "AI Generated",
    "module3_score" : 50.0,
    "composite_score": 75.0,
    "risk_level"    : "High Risk",
    "behavior_label": "Anomaly",
    "grade_jump"    : 5.0,
    "g1"            : 8,
    "g2"            : 7,
    "g3"            : 18,
    "absences"      : 1,
    "studytime"     : 2,
    "failures"      : 0,
    "essay_text"    : "Test essay",
    "faculty_note"  : "",
    "analyzed_at"   : "2026-05-06 10:00:00"
    }).execute()
    print("Insert successful!")
    print("Response:", response.data)

    # test fetch
    response2 = client.table("risk_results").select("*").execute()
    print("Fetch successful!")
    print("Total records:", len(response2.data))

except Exception as e:
    print("ERROR:", str(e))

# test update note
from app.database import get_student_history, update_note

history = get_student_history("TEST001")
print("History found:", len(history), "records")
if history:
    print("Record ID:", history[0]["id"])
    print("ID type:", type(history[0]["id"]))
    result = update_note(history[0]["id"], "This is a test note")
    print("Update result:", result)

    # verify
    history2 = get_student_history("TEST001")
    print("Note after update:", history2[0].get("faculty_note"))