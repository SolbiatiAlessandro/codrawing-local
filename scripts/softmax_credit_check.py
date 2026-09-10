"""Read-only credit API inspection using the existing CI-held login.

Does not create an episode or print authentication material.
"""
import json
import re
from urllib.parse import urljoin, urlparse

from coworld.api_client import CoworldApiClient


def main():
    with CoworldApiClient.from_login(server_url="https://softmax.com/api") as client:
        spec = client._get("/openapi.json", dict)
        Path = __import__("pathlib").Path
        Path("public-openapi.json").write_text(json.dumps(spec))
        print("PUBLIC_PATHS", json.dumps(list(spec["paths"])), flush=True)
        who = client._get("/whoami", dict)
        print("AUTH_SUBJECT", who.get("subject_type"), flush=True)
        print("WHOAMI_KEYS", json.dumps(list(who)), flush=True)
        for key, value in who.items():
            if any(word in key.lower() for word in ("credit", "balance", "allowance")):
                print("ALLOWANCE", key, json.dumps(value), flush=True)
        for name, schema in spec.get("components", {}).get("schemas", {}).items():
            if any(word in json.dumps(schema).lower() for word in ("credit", "balance", "allowance")):
                print("CREDIT_SCHEMA", name, json.dumps(schema), flush=True)
        # Inspect public frontend code for account APIs missing from public OpenAPI.
        page = client._http_client.get("https://softmax.com/observatory/v2")
        print("PUBLIC_PAGE_STATUS", page.status_code, flush=True)
        if page.status_code == 200:
            scripts = re.findall(r'<script[^>]+src="([^"]+)"', page.text)
            for src in scripts[:30]:
                url = urljoin("https://softmax.com", src)
                if urlparse(url).hostname != "softmax.com":
                    continue
                response = client._http_client.get(url)
                if response.status_code != 200:
                    continue
                for match in re.finditer(r"credits|creditBalance|credit_balance|estimateCost|estimate-cost", response.text):
                    print("PUBLIC_JS_CREDIT", src, response.text[max(0, match.start()-160):match.end()+240], flush=True)
        names = set()
        for path, operations in spec["paths"].items():
            if any(word in path.lower() for word in ("credit", "estimate", "pricing", "balance")):
                print("API", path, json.dumps(operations), flush=True)
                # Only definition data; resolve referenced public schemas below.
                def collect(value):
                    if isinstance(value, dict):
                        if "$ref" in value:
                            names.add(value["$ref"].rsplit("/", 1)[-1])
                        for child in value.values():
                            collect(child)
                    elif isinstance(value, list):
                        for child in value:
                            collect(child)
                collect(operations)
        for name in sorted(names):
            print("SCHEMA", name, json.dumps(spec.get("components", {}).get("schemas", {}).get(name)), flush=True)
        print("Read-only API discovery complete; no episode created.", flush=True)


if __name__ == "__main__":
    main()
