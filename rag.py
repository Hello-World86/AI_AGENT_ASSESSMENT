# Here, we perform policy retrieval using vector store search

import logging
from vector_store import search

logger = logging.getLogger(__name__)


def retrieve_policy(query: str) -> dict:
    """Retrieve relevant policies based on the query."""
    try:
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided to retrieve_policy")
            return {"error": "Query must be a non-empty string"}

        logger.debug(f"Retrieving policy for query: {query}")

        results = search(query, k=3)

        if "error" in results:
            logger.error(f"Error in search results: {results['error']}")
            return {"error": results["error"]}

        logger.info(f"Policy retrieval successful for query: {query}")
        return results

    except Exception as e:
        logger.error(f"Error retrieving policy: {str(e)}", exc_info=True)
        return {"error": f"Error retrieving policy: {str(e)}"}
