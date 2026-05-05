# Database helper for AcademicGuard
import sqlite3
import os
from datetime import datetime

DB_PATH = "database/results.db"
os.makedirs("database", exist_ok=True)

def init_db():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id      TEXT,
            student_name    TEXT,
            module1_score   REAL,
            module2_score   REAL,
            module2_label   TEXT,
            module3_score   REAL,
            composite_score REAL,
            risk_level      TEXT,
            behavior_label  TEXT,
            grade_jump      REAL,
            G1              INTEGER,
            G2              INTEGER,
            G3              INTEGER,
            absences        INTEGER,
            studytime       INTEGER,
            failures        INTEGER,
            essay_text      TEXT,
            faculty_note    TEXT,
            analyzed_at     TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_result(result):
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO risk_results (
            student_id, student_name, module1_score, module2_score,
            module2_label, module3_score, composite_score, risk_level,
            behavior_label, grade_jump, G1, G2, G3,
            absences, studytime, failures, essay_text, analyzed_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        result["student_id"],
        result["student_name"],
        result["module1_ai_score"],
        result["module2_style_score"],
        result.get("module2_label", ""),
        result["module3_behavior_score"],
        result["composite_score"],
        result["risk_level"],
        result["behavior_label"],
        result["grade_jump"],
        result["G1"], result["G2"], result["G3"],
        result["absences"], result["studytime"], result["failures"],
        result.get("essay_text", ""),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def update_note(record_id, note):
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE risk_results SET faculty_note=? WHERE id=?",
        (note, record_id)
    )
    conn.commit()
    conn.close()

def get_all_results():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, student_id, student_name, module1_score, module2_score,
               module2_label, module3_score, composite_score, risk_level,
               behavior_label, grade_jump, G1, G2, G3,
               absences, studytime, failures, faculty_note, analyzed_at
        FROM risk_results
        ORDER BY analyzed_at DESC
    """)
    rows    = cursor.fetchall()
    columns = [d[0] for d in cursor.description]
    conn.close()
    return [dict(zip(columns, row)) for row in rows]

def get_student_history(student_id):
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM risk_results
        WHERE student_id=?
        ORDER BY analyzed_at DESC
    """, (student_id,))
    rows    = cursor.fetchall()
    columns = [d[0] for d in cursor.description]
    conn.close()
    return [dict(zip(columns, row)) for row in rows]

def delete_record(record_id):
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM risk_results WHERE id=?", (record_id,))
    conn.commit()
    conn.close()

# initialize on import
init_db()