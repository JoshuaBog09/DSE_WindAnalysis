import requests
import h5py

import logging
import sys

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

API_KEY = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6ImE1OGI5NGZmMDY5NDRhZDNhZjFkMDBmNDBmNTQyNjBkIiwiaCI6Im11cm11cjEyOCJ9"
URL = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/radar_forecast/versions/1.0/files"

res = requests.get(URL, headers={"Authorization": API_KEY}).json()
datafiles = res.get("files")
filename = datafiles[0]["filename"]

endpoint = f"https://api.dataplatform.knmi.nl/open-data/v1/datasets/radar_forecast/versions/1.0/files/{filename}/url"

res_file = requests.get(endpoint, headers={"Authorization": API_KEY}).json()

download_url = res_file.get("temporaryDownloadUrl")

try:
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
except Exception:
    logger.exception("Unable to download file using download URL")
    sys.exit(1)

logger.info(f"Successfully downloaded dataset file to {filename}")

with h5py.File("RAD_NL25_PCP_FM_201910261400.h5", "r") as f:
    # Print all root level object names (aka keys)
    # these can be group or dataset names
    print("Keys: %s" % f.keys())