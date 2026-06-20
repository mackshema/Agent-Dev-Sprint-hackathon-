import os
import re
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# -- Credentials ---------------------------------------------------------------
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
ARENA_ID_TOKEN  = os.getenv("ARENA_ID_TOKEN")
MCP_ENDPOINT    = os.getenv("MCP_ENDPOINT", "https://agent-arena-623774504237.asia-southeast1.run.app/mcp")

# -- Agent Identity ------------------------------------------------------------
_raw_name   = os.getenv("AGENT_NAME", "ApexAgent_v1")
AGENT_NAME  = re.sub(r'[^a-zA-Z0-9_]', '_', _raw_name)
if AGENT_NAME and not (AGENT_NAME[0].isalpha() or AGENT_NAME[0] == '_'):
    AGENT_NAME = '_' + AGENT_NAME

AGENT_STACK = os.getenv("AGENT_STACK", "Python / ADK / Gemini 2.5 Multi-Agent")

# -- Multi-Model Strategy ------------------------------------------------------
# Phase 1 (Classify) — cheapest model
MODEL_CLASSIFY = os.getenv("MODEL_CLASSIFY", "gemini-2.5-flash-lite")
# Phase 2 (Plan), Phase 5 (Critique), Phase 6 (Judge) — fast model
MODEL_PLAN     = os.getenv("MODEL_PLAN",     "gemini-2.5-flash")
MODEL_CRITIQUE = os.getenv("MODEL_CRITIQUE", "gemini-2.5-flash")
MODEL_JUDGE    = os.getenv("MODEL_JUDGE",    "gemini-2.5-flash")
# Phase 4 (Solve) — best available model (pro requires paid tier; flash works on free)
MODEL_SOLVE    = os.getenv("MODEL_SOLVE",    "gemini-2.5-flash")
# Fallback / default
MODEL          = os.getenv("MODEL",          "gemini-2.5-flash")

# Ordered fallback chain for the solver — tried left-to-right on quota errors
SOLVE_FALLBACK_CHAIN = [
    MODEL_SOLVE,
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

# -- Loop Control --------------------------------------------------------------
try:
    MAX_TURNS = int(os.getenv("MAX_TURNS", "20"))
except ValueError:
    MAX_TURNS = 20

# -- Gemini 2.5 Flash pricing (approximate) -----------------------------------
# Flash:      $0.075 / 1M input,  $0.30 / 1M output
# Flash-Lite: $0.019 / 1M input,  $0.075 / 1M output
# Pro:        $1.25  / 1M input,  $5.00 / 1M output  (≤200k ctx)
PRICING = {
    "gemini-2.5-pro":        (1.25,  5.00),
    "gemini-2.5-flash":      (0.075, 0.30),
    "gemini-2.5-flash-lite": (0.019, 0.075),
    "gemini-2.0-flash":      (0.075, 0.30),
}

def get_model_pricing(model: str) -> tuple[float, float]:
    """Returns (input_price_per_1M, output_price_per_1M) for a model."""
    for key in PRICING:
        if key in model:
            return PRICING[key]
    return (0.075, 0.30)  # fallback to flash pricing


def validate_config(exit_on_failure: bool = True) -> bool:
    """Validate project configuration variables."""
    errors = []

    if not GEMINI_API_KEY:
        errors.append(
            "GEMINI_API_KEY is not set.\n"
            "  -> Please obtain a free API key from https://aistudio.google.com and add it to your .env file."
        )

    if not ARENA_ID_TOKEN or ARENA_ID_TOKEN.startswith("EYJH"):
        errors.append(
            "ARENA_ID_TOKEN is missing or is uppercased (invalid JWT).\n"
            "  -> JWT tokens are case-sensitive. Open the Arena web app, go to DevTools -> Application ->\n"
            "     Storage -> Local Storage, copy the ID_TOKEN value EXACTLY (do not uppercase it),\n"
            "     and paste it into your .env file."
        )

    if errors:
        print("\n================ CONFIGURATION ERROR ================", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
        print("=====================================================\n", file=sys.stderr)
        if exit_on_failure:
            sys.exit(1)
        return False

    return True
