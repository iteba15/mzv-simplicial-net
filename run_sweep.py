"""
Long-running sweep script. Run with:
    python run_sweep.py zagier
    python run_sweep.py apery
    python run_sweep.py ramanujan
"""
import sys
import json
import logging
import pathlib
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sweep.log", encoding="utf-8"),
    ],
)

from zeta_hunter.pcf.families import APERY, ZAGIER, RAMANUJAN
from zeta_hunter.pcf.sweeper import PCFSweeper
from zeta_hunter.logger import RunLogger

FAMILIES = {"apery": APERY, "zagier": ZAGIER, "ramanujan": RAMANUJAN}

name = sys.argv[1].lower() if len(sys.argv) > 1 else "zagier"
family = FAMILIES[name]

RUN_ID = f"sweep_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logging.info("Starting %s sweep | run_id=%s | combinations=%s",
             family.name, RUN_ID, f"{family.total_combinations:,}")

logger = RunLogger(RUN_ID)
sweeper = PCFSweeper(family, logger)
sweeper.sweep()

data = json.loads(pathlib.Path(f"runs/{RUN_ID}.json").read_text())
stats = data["stats"]
logging.info(
    "Done | scanned=%s | stage1=%s | stage2=%s | elapsed=%.1fs | throughput=%s PCFs/hr",
    f"{stats['total_scanned']:,}",
    stats["stage1_hits"],
    stats["stage2_hits"],
    stats["elapsed_seconds"],
    f"{stats['throughput_per_hour']:,}",
)

hits = data["stage2_hits"]
if hits:
    logging.info("Stage 2 hits:")
    for h in hits:
        logging.info("  [%s] target=%s precision=%.1fd a=%s b=%s",
                     h["verdict"], h["target"], h["stage2_precision_digits"],
                     h["a_coeffs"], h["b_coeffs"])
