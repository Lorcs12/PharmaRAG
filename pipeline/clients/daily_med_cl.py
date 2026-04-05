from config import CFG
from typing import Optional
import json
import time
import urllib.request
from logger import get_logger, Timer
from ..helpers.utils import strip_html

log = get_logger("dailymed", CFG.log.file, CFG.log.level)

class DailyMedClient:
    """
    Wraps FDA DailyMed REST API v2.

    Rate limit: 240 requests/min for anonymous users.
    Our delay: 0.34s between calls = ~176/min — comfortably under limit.
    """

    BASE = CFG.api.dailymed_base
    OPENFDA = CFG.api.openfda_base

    def _get(self, url: str, retries: int = 3) -> Optional[dict]:
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"Accept": "application/json",
                             "User-Agent": "PharmaRAG/1.0 research@pharmarag.ai"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = CFG.api.retry_delay * (attempt + 1)
                    log.warning(f"Rate limited — waiting {wait}s (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                elif e.code in (500, 502, 503):
                    time.sleep(CFG.api.retry_delay)
                else:
                    log.error(f"HTTP {e.code} for {url}: {e}")
                    return None
            except Exception as e:
                log.error(f"Request failed: {e}")
                if attempt < retries - 1:
                    time.sleep(CFG.api.retry_delay)
        return None

    def search_drug(self, drug_name: str) -> Optional[dict]:
        query = f'openfda.generic_name:"{drug_name}"'
        url = (f"{self.OPENFDA}?search={urllib.parse.quote(query)}"
               f"&limit=1")
        time.sleep(CFG.api.request_delay)
        return self._get(url)

    def get_label_by_generic_name(self, drug_name: str) -> Optional[dict]:
        raw_query = f'openfda.generic_name:"{drug_name}" AND openfda.is_original_packager:true AND openfda.application_number:(NDA* OR ANDA*)'
        
        encoded_query = urllib.parse.quote(raw_query)
        
        url = f"{self.OPENFDA}?search={encoded_query}&limit=1&sort=effective_time:desc"
        
        time.sleep(CFG.api.request_delay)
        return self._get(url)

    def get_spl_by_set_id(self, set_id: str) -> Optional[dict]:
        url = f"{self.BASE}/spls/{set_id}.json"
        time.sleep(CFG.api.request_delay)
        return self._get(url)
    
    def get_interactions(self, drug_name: str) -> list[dict]:
        query = f'openfda.generic_name:"{drug_name}"'
        url = (f"{self.OPENFDA}?search={urllib.parse.quote(query)}"
            f"&limit=1&sort=effective_time:desc")
        time.sleep(CFG.api.request_delay)
        data = self._get(url)
        if not data:
            return []
        results = data.get("results", [])
        if not results:
            return []
        interactions = []
        for section_text in results[0].get("drug_interactions", []):
            clean = strip_html(section_text)
            if clean:
                interactions.append({
                    "interacting_rxcui": None,
                    "severity":          None,
                    "description":       clean,
                })
        return interactions