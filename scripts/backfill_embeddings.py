"""Call the /knowledge/backfill-embeddings endpoint in a loop until done."""

import sys
import time
import requests

API_BASE = "https://stat-research-assistant-pfemb.ondigitalocean.app/api"
USERNAME = "admin"
PASSWORD = "admin123"
BATCH_SIZE = 100


def login() -> str:
    resp = requests.post(
        f"{API_BASE}/auth/login",
        json={"username": USERNAME, "password": PASSWORD},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def backfill_batch(token: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/knowledge/backfill-embeddings",
        params={"batch_size": BATCH_SIZE},
        headers={"Authorization": f"Bearer {token}"},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    print("Logging in...")
    token = login()

    round_num = 0
    total_processed = 0
    while True:
        round_num += 1
        print(f"\n--- Round {round_num} (batch_size={BATCH_SIZE}) ---")
        result = backfill_batch(token)
        processed = result["processed"]
        remaining = result["remaining"]
        total = result["total"]
        total_processed += processed
        print(f"  Processed: {processed}, Remaining: {remaining}, Total: {total}")
        print(f"  Cumulative: {total_processed} embeddings generated")

        if remaining == 0:
            print(f"\nDone! All {total} knowledge units have embeddings.")
            break

        # Small delay to avoid overwhelming the server
        time.sleep(1)


if __name__ == "__main__":
    main()
