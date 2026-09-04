from he_utils import create_context, encrypt_vector, decrypt_vector

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling/staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself, or that you are a failure",
    "Trouble concentrating on things",
    "Moving/speaking slowly, or being fidgety/restless",
    "Thoughts that you would be better off dead, or of self-harm"
]

def severity_band(score):
    if score <= 4:
        return "None-minimal"
    elif score <= 9:
        return "Mild"
    elif score <= 14:
        return "Moderate"
    elif score <= 19:
        return "Moderately severe"
    else:
        return "Severe"

def run_encrypted_phq9(answers):
    """
    answers: list of 9 integers, each 0-3, matching PHQ9_QUESTIONS order.
    Returns the decrypted total score (0-27) and its severity band.
    The scoring itself (a simple sum) happens while the answers are encrypted —
    the server computing this never sees the individual raw answers.
    """
    if len(answers) != 9:
        raise ValueError("PHQ-9 requires exactly 9 answers")

    context = create_context()
    encrypted_answers = encrypt_vector(context, [float(a) for a in answers])

    # weights are all 1 — we're just summing, not weighting any question higher
    weights = [1.0] * 9
    encrypted_total = encrypted_answers.dot(weights)

    decrypted_score = decrypt_vector(encrypted_total)
    total_score = round(decrypted_score[0] if isinstance(decrypted_score, list) else decrypted_score)

    return total_score, severity_band(total_score)

if __name__ == "__main__":
    # test example: moderate-ish set of answers
    sample_answers = [2, 2, 1, 2, 0, 1, 1, 0, 0]
    score, band = run_encrypted_phq9(sample_answers)
    print(f"PHQ-9 encrypted score: {score}/27 — {band}")