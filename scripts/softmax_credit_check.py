"""Check the current user's Softmax allowance without launching an episode."""
import json
from coworld.api_client import CoworldApiClient


def main():
    with CoworldApiClient.from_login(server_url="https://softmax.com/api") as client:
        for path in ("/v2/credits/me", "/v2/credits", "https://softmax.com/api/credits"):
            r = client._http_client.get(path, headers=client._headers())
            print("ACCOUNT_ROUTE", path, r.status_code, flush=True)
            if r.status_code == 200:
                data = r.json()
                print("ACCOUNT_KEYS", list(data) if isinstance(data, dict) else type(data).__name__, flush=True)
                def numeric(value, prefix=""):
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if any(word in key.lower() for word in ("token", "secret", "email", "user", "name", "id")):
                                continue
                            numeric(item, prefix + key + ".")
                    elif isinstance(value, (int, float, bool)) or value is None:
                        print("ACCOUNT_VALUE", prefix, value, flush=True)
                numeric(data)
        r = client._http_client.get("https://softmax.com/play.md")
        print("PLAY_DOC_STATUS", r.status_code, flush=True)
        if r.status_code == 200:
            text = r.text
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if any(word in line.lower() for word in ("credit", "balance", "budget", "account")):
                    print("PLAY_DOC", "\n".join(lines[max(0,idx-2):idx+3]), flush=True)
        print("No episode created.", flush=True)


if __name__ == "__main__":
    main()
