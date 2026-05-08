from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def check_plagiarism(current_essay, past_records, current_student_id, threshold=0.8):

    essays = []
    student_ids = []

    for r in past_records:
        if r.get("essay_text") and str(r.get("student_id")) != str(current_student_id):
            essays.append(r["essay_text"])
            student_ids.append(r["student_id"])

    if len(essays) == 0:
        return {
            "is_copied": False,
            "similarity": 0,
            "matched_student": None
        }

    # Add current essay at end
    essays.append(current_essay)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(essays)

    similarity_matrix = cosine_similarity(tfidf[-1], tfidf[:-1])

    max_sim = similarity_matrix.max()

    if max_sim >= threshold:
        idx = similarity_matrix.argmax()
        return {
            "is_copied": True,
            "similarity": round(max_sim * 100, 2),
            "matched_student": student_ids[idx]
        }

    return {
        "is_copied": False,
        "similarity": round(max_sim * 100, 2),
        "matched_student": None
    }