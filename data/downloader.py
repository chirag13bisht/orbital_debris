"""
data/downloader.py
Downloads Cosmos-2251 collision debris TLEs from Space-Track.org.
Run this first: python -m data.downloader
"""

import os, json, time, requests
from datetime import datetime, timedelta
from pathlib import Path

import config

class SpaceTrackClient:
    """Authenticated client for Space-Track.org REST API."""

    LOGIN_URL   = f"{config.SPACETRACK_BASE_URL}/ajaxauth/login"
    BASE_URL    = f"{config.SPACETRACK_BASE_URL}/basicspacedata/query"

    def __init__(self):
        self.session  = requests.Session()
        self.logged_in = False

    def login(self):
        resp = self.session.post(self.LOGIN_URL, data={
            "identity": config.SPACETRACK_USER,
            "password": config.SPACETRACK_PASSWORD,
        }, timeout=30)
        if resp.status_code != 200 or "login" in resp.url:
            raise ConnectionError(f"Space-Track login failed: {resp.status_code}")
        self.logged_in = True
        print("✅ Logged in to Space-Track.org")

    def logout(self):
        self.session.get(
            f"{config.SPACETRACK_BASE_URL}/ajaxauth/logout", timeout=10
        )
        self.logged_in = False

    def _get(self, url, params=None):
        if not self.logged_in:
            self.login()
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp

    # ── Query builders ────────────────────────────────────────────

    def get_cosmos_debris_norad_ids(self):
        """
        Fetch NORAD IDs of all catalogued objects from the Cosmos-2251
        collision (INTLDES starting with '09005' = 2009-005).
        """
        url = (
            f"{self.BASE_URL}/class/satcat"
            f"/INTLDES/~~09005"        # Cosmos-2251 fragments
            f"/OBJECT_TYPE/DEBRIS"
            f"/CURRENT/Y"
            f"/orderby/NORAD_CAT_ID asc"
            f"/format/json"
            f"/emptyresult/show"
        )
        resp = self._get(url)
        data = resp.json()
        ids  = [int(obj["NORAD_CAT_ID"]) for obj in data]
        print(f"Found {len(ids)} Cosmos-2251 debris objects in SATCAT")
        return ids

    def get_tles_for_object(self, norad_id: int,
                             epoch_start: str, epoch_end: str):
        """
        Fetch all TLEs for one NORAD ID between two epoch dates.
        Returns list of dicts with keys: EPOCH, MEAN_MOTION, ECCENTRICITY,
        INCLINATION, RA_OF_ASC_NODE, ARG_OF_PERICENTER, MEAN_ANOMALY,
        BSTAR, SEMIMAJOR_AXIS, PERIOD, OBJECT_ID, NORAD_CAT_ID.
        """
        url = (
            f"{self.BASE_URL}/class/gp_history"
            f"/NORAD_CAT_ID/{norad_id}"
            f"/EPOCH/{epoch_start}--{epoch_end}"
            f"/orderby/EPOCH asc"
            f"/format/json"
            f"/emptyresult/show"
        )
        resp = self._get(url)
        return resp.json()


def download_all_tles(save_dir: str = "data/raw") -> str:
    """
    Main download function. Fetches all Cosmos-2251 debris TLEs
    and saves as JSON files, one per NORAD ID.
    Returns path to save directory.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = os.path.join(save_dir, "manifest.json")

    # Check if already downloaded
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Found existing download: {manifest['n_objects']} objects, "
              f"downloaded {manifest['download_date']}")
        return save_dir

    client = SpaceTrackClient()

    try:
        client.login()

        # Step 1: Get NORAD IDs
        norad_ids = client.get_cosmos_debris_norad_ids()

        if not norad_ids:
            print("No objects found — falling back to known ID range")
            # Known approximate range for Cosmos-2251 debris
            norad_ids = list(range(33594, 36000))

        # Step 2: Download TLEs for each object
        successful, skipped = 0, 0
        all_ids_downloaded  = []

        for i, nid in enumerate(norad_ids):
            out_file = os.path.join(save_dir, f"{nid}.json")

            if os.path.exists(out_file):
                all_ids_downloaded.append(nid)
                successful += 1
                continue

            tles = client.get_tles_for_object(
                nid,
                config.COSMOS_DEBRIS_EPOCH_START,
                config.COSMOS_DEBRIS_EPOCH_END
            )

            if len(tles) < config.MIN_RECORDS_PER_OBJECT:
                skipped += 1
            else:
                with open(out_file, "w") as f:
                    json.dump(tles, f)
                all_ids_downloaded.append(nid)
                successful += 1

            # Progress
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(norad_ids)} processed, "
                      f"{successful} saved, {skipped} skipped")

            # Rate-limit: be polite to Space-Track servers
            time.sleep(0.5)

        # Save manifest
        manifest = {
            "n_objects":     successful,
            "norad_ids":     all_ids_downloaded,
            "epoch_start":   config.COSMOS_DEBRIS_EPOCH_START,
            "epoch_end":     config.COSMOS_DEBRIS_EPOCH_END,
            "download_date": datetime.now().isoformat(),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n✅ Download complete: {successful} objects saved to '{save_dir}'")
        return save_dir

    finally:
        client.logout()


if __name__ == "__main__":
    download_all_tles()
