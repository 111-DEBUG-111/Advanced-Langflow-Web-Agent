from dotenv import load_dotenv
import os
import requests

load_dotenv()

def serp_search(query: str, engine: str = "google") -> str:
    """
    Searches the web using the Firecrawl API.
    Returns a formatted string containing high-quality markdown snippets.
    """
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return "Error: FIRECRAWL_API_KEY is not set in the environment."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        "limit": 5,
        "scrapeOptions": {
            "formats": ["markdown"],
            "onlyMainContent": True
        }
    }

    try:
        response = requests.post(
            "https://api.firecrawl.dev/v2/search",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        res_data = response.json()

        if not res_data.get("success"):
            return f"Error: Firecrawl search was unsuccessful: {res_data}"

        data_obj = res_data.get("data", {})
        results = []
        if isinstance(data_obj, list):
            results = data_obj
        elif isinstance(data_obj, dict):
            results = data_obj.get("web", [])

        formatted_results = []

        for idx, item in enumerate(results, 1):
            title = item.get("title", "No Title")
            url = item.get("url", "No URL")
            description = item.get("description", "")
            markdown = item.get("markdown", "")
            
            # Truncate markdown snippet to preserve token context window
            snippet = markdown[:1200] + "..." if len(markdown) > 1200 else markdown
            
            formatted_results.append(
                f"Result {idx}:\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Description: {description}\n"
                f"Content:\n{snippet}\n"
                f"{'-'*40}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Error executing Firecrawl search: {e}"