import streamlit as st
import pandas as pd
from config import test_url_scrapability

def test_category_urls(urls_dict):
    """
    Test all URLs in the dictionary and return a filtered dictionary
    with only scrapable URLs.

    Args:
        urls_dict (dict): Dictionary of URLs by category

    Returns:
        dict: Filtered dictionary with only scrapable URLs
    """
    filtered_urls = {}
    results = []

    with st.spinner("Testing URLs for scraping compatibility..."):
        for category, urls in urls_dict.items():
            filtered_urls[category] = []

            for url in urls:
                is_scrapable, content_size = test_url_scrapability(url)
                results.append({
                    "Category": category,
                    "URL": url,
                    "Scrapable": is_scrapable,
                    "Content Size (chars)": content_size
                })

                if is_scrapable and content_size > 500:  # Ensure there's meaningful content
                    filtered_urls[category].append(url)

    # Show results table
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # Summary
    total_urls = len(results)
    scrapable_urls = sum(1 for r in results if r["Scrapable"] and r["Content Size (chars)"] > 500)
    st.info(f"Found {scrapable_urls} scrapable URLs out of {total_urls} total URLs.")

    return filtered_urls

def get_all_scrapable_urls(urls_dict):
    """
    Get a flat list of all scrapable URLs from all categories.

    Args:
        urls_dict (dict): Dictionary of scrapable URLs by category

    Returns:
        list: Flat list of all scrapable URLs
    """
    return [url for category_urls in urls_dict.values() for url in category_urls]
