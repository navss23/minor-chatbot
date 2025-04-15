import os
from dotenv import load_dotenv

# Load environment variables
def load_api_key():
    """
    Load Google API key from environment variables.

    Returns:
        str: Google API key
    """
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".env"))
    load_dotenv(env_path)
    return os.getenv("GOOGLE_API_KEY")

# URLs for document loading
URLS = [
    "https://www.careerindia.com/courses/after-12th-science-courses-career-options-explained-024108.html",
    "https://www.successcds.net/Career/qna/career-options-after-12th.html",
    "https://byjus.com/commerce/career-options-after-12th-commerce/"
]

# Predefined category questions
CATEGORY_QUESTIONS = {
    "Engineering & Technology": ["What are the best engineering fields for the future?", "Is AI engineering a good career?"],
    "Medical & Healthcare": ["Which medical careers don't require NEET?", "What are the top paramedical courses?"],
    "Business & Management": ["What career options exist in finance?", "Is an MBA worth it in 2025?"],
    "Arts & Humanities": ["How to build a career in journalism?", "What are the best career options in humanities?"],
    "Government & Civil Services": ["How to join the Indian Army after 12th?", "What are the best government exams after 12th?"]
}
