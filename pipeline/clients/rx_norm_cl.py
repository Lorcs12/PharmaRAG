
from config import CFG
from typing import Optional
import json
import time
import urllib.request
from logger import get_logger


log = get_logger("rxnorm", CFG.log.file, CFG.log.level)
class RxNormClient:
    BASE = CFG.api.rxnorm_base

    def _get(self, url: str) -> Optional[dict]:
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            log.warning(f"RxNorm request failed: {e}")
            return None

    def get_rxcui(self, drug_name: str) -> Optional[str]:
        """Resolve drug name → rxcui (canonical drug concept ID)."""
        url = f"{self.BASE}/rxcui.json?name={urllib.parse.quote(drug_name)}&search=1"
        time.sleep(0.2)
        data = self._get(url)
        if data:
            rxcui = data.get("idGroup", {}).get("rxnormId", [])
            if rxcui:
                return rxcui[0]
        return None

    def get_atc_code(self, rxcui: str) -> Optional[str]:
        """Get ATC classification code for a rxcui."""
        url = f"{self.BASE}/rxclass/class/byRxcui.json?rxcui={rxcui}&relaSource=ATC"
        time.sleep(0.2)
        data = self._get(url)
        if data:
            classes = data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
            # Return level-4 ATC code (most specific before individual drug level)
            for c in classes:
                code = c.get("rxclassMinConceptItem", {}).get("classId", "")
                if len(code) == 5:  # e.g. C10AA
                    return code
            if classes:
                return classes[0].get("rxclassMinConceptItem", {}).get("classId", "")
        return None