from googleapiclient import discovery
import json

import logging

logging.getLogger("googleapicliet.discovery_cache").setLevel(logging.ERROR)
API_KEY = "Perspective_API_KEY"



def tox_classifier(text: str) -> float:
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
        cache_discovery=False,
    )

    analyze_request = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {"TOXICITY": {}},
    }

    response = client.comments().analyze(body=analyze_request).execute()
    return float(response["attributeScores"]["TOXICITY"]["summaryScore"]["value"])


if __name__ == "__main__":
    text = "this is f*cking bullsh*t"

    out = tox_classifier(text)
    print(out)
