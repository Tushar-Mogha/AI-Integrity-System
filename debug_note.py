# save this as debug_note.py and run it
import os
from dotenv import load_dotenv
load_dotenv()

from app.database import get_student_history, update_note, get_all_results

# step 1 - get all records
print("=== All Records ===")
all_records = get_all_results()
print(f"Total records: {len(all_records)}")
for r in all_records[:3]:
    print(f"ID: {r['id']} | Student: {r['student_name']} | Note: '{r.get('faculty_note', 'MISSING')}'")

# step 2 - get history for a specific student
# replace STU001 with the actual student ID you tested with
student_id = input("\nEnter the student ID you tested with: ")
history = get_student_history(student_id)
print(f"\nHistory for {student_id}: {len(history)} records")
if history:
    print(f"Latest record ID: {history[0]['id']}")
    print(f"Current note: '{history[0].get('faculty_note', 'MISSING')}'")

    # step 3 - try updating note
    record_id = history[0]['id']
    print(f"\nTrying to update note for record ID: {record_id}")
    result = update_note(record_id, "DEBUG TEST NOTE")
    print(f"Update result: {result}")

    # step 4 - verify
    history2 = get_student_history(student_id)
    print(f"Note after update: '{history2[0].get('faculty_note', 'MISSING')}'")
else:
    print("No history found for this student!")
    print("Available student IDs:")
    for r in all_records:
        print(f"  - {r['student_id']} : {r['student_name']}")