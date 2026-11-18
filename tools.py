import urllib.parse
import requests
from langchain_core.tools import tool


@tool
def wiki_summary(topic: str) -> str:
    """
    Get a short factual summary of a topic from Wikipedia.
    Uses the free Wikipedia REST API — no API key.
    """
    cleaned_topic = topic.strip().replace(" ", "_")
    title = urllib.parse.quote(cleaned_topic)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    headers = {
        "User-Agent": "SupervisorAgent/1.0 (educational-exercise; mailto:contact@example.com)"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            extract = data.get("extract")
            if extract:
                return extract
            return f"No extract available for '{topic}'."
        else:
            return f"Could not find a Wikipedia summary for '{topic}'. Status code: {response.status_code}"
    except Exception as e:
        return f"Error fetching Wikipedia summary for '{topic}': {e}"
