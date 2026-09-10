"""Read-only credit API inspection using the existing CI-held login.

Does not create an episode or print authentication material.
"""
import json

from coworld.api_client import CoworldApiClient


def main():
    with CoworldApiClient.from_login(server_url="https://softmax.com/api") as client:
        spec = client._get("/openapi.json", dict)
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
