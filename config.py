import os
import requests
from bs4 import BeautifulSoup
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

# Updated URLs for document loading by category
URLS = {
    "Engineering & Technology": [
        "https://engineering.saraswatikharghar.edu.in/engineering-courses-in-india/",
        "https://engineering.lingayasvidyapeeth.edu.in/blog/top-career-options-after-engineering/",
        "https://www.bmu.edu.in/social/engineering-courses-in-future-with-salary-and-scope/"
    ],
    "Medical & Healthcare": [
        "https://care.edu.in/blog/top-medical-courses-in-india-course-details-and-scope/",
        "https://lloydpharmacy.edu.in/blog/best-medical-courses.html",
        "https://msu.edu.in/wise/what-are-the-career-options-if-you-do-not-clear-the-neet-exam-2023/"
    ],
    "Business & Management": [
        "https://srbs.edu.in/blogs/business-management-career-guide/",
        "https://www.lloydbusinessschool.edu.in/blog/top-career-options-after-bba.html",
        "https://www.msu.edu.in/blog/build-your-career-in-business-management-education-opportunities-and-salary-insights"
    ],
    "Arts & Humanities": [
        "https://jgu.edu.in/blog/2023/12/29/career-in-liberal-arts/",
        "https://inspiria.edu.in/best-career-options-for-arts-students-after-class-12/",
        "https://rvu.edu.in/life-changing-career-paths-in-liberal-arts-and-sciences/"
    ],
    "Government & Civil Services": [
        "https://dge.gov.in/dge/nics/introduction",
        "https://lakshadweep.gov.in/notice/interaction-with-civil-service-aspirants-on-career-guidance-civil-services-as-an-option-for-better-future/",
        "https://www.employmentnews.gov.in/career_guide.asp"
    ]
}

# For backward compatibility with the original code structure
# Flatten the URLs for functions that expect a simple list
ALL_URLS = [url for category_urls in URLS.values() for url in category_urls]

# Function to test if a URL is scrapable
def test_url_scrapability(url, timeout=5):
    """
    Test if a URL can be scraped successfully.

    Args:
        url (str): URL to test
        timeout (int): Request timeout in seconds

    Returns:
        tuple: (is_scrapable (bool), content_size (int))
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Check if the page has meaningful content (at least some paragraphs)
            paragraphs = soup.find_all('p')
            content_size = sum(len(p.get_text()) for p in paragraphs)
            return True, content_size
        return False, 0
    except Exception as e:
        return False, 0

# Updated category questions to better match content in scrape-friendly URLs
CATEGORY_QUESTIONS = {
    "Engineering & Technology": [
        "What are the top engineering courses available in India?",
        "What career options do I have after completing an engineering degree?",
        "Which engineering fields will have the best scope in the future?",
        "What are the highest paying engineering jobs in India?"
    ],
    "Medical & Healthcare": [
        "What medical courses can I pursue in India?",
        "What are my options if I don't qualify for NEET?",

        "Which medical specializations have the highest demand?",
        "What are alternative healthcare career paths besides MBBS?"
    ],
    "Business & Management": [
        "What career paths can I follow after completing a BBA?",
        "What are the top management specializations in India?",
        "How does a business management degree help in career growth?",
        "What are the salary prospects with a business management degree?",
        "What skills should I develop for a career in business management?"
    ],
    "Arts & Humanities": [
        "What career options are available after completing an Arts degree?",
        "How can I build a career in liberal arts?",
        "What are the advantages of studying humanities?",
        "Which Arts specializations have good job prospects?",
        "How can I prepare for a career in social sciences?"
    ],
    "Government & Civil Services": [
        "What are the different career paths in civil services?",
        "How can I prepare for government service exams?",
        "What are the eligibility criteria for civil service exams?",
        "What career opportunities exist in government sectors?",
        "How can I improve my chances of clearing civil service exams?"
    ]
}
