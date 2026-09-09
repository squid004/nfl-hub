"""Configuration from environment variables only (Actions `env:` in CI, `.env` for local dev).

No config.toml in v2 — the deploy target is GitHub Actions, which passes everything as env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OAUTH_PATH = Path(os.environ.get("YAHOO_OAUTH_PATH", ROOT / "oauth2.json"))


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _floats(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [float(x) for x in raw.replace(" ", "").split(",") if x]


@dataclass
class Supabase:
    url: str
    anon_key: str

    @property
    def rest(self) -> str:
        return f"{self.url.rstrip('/')}/rest/v1"


@dataclass
class YahooCfg:
    league_id: str
    team_key: str
    enabled: bool


@dataclass
class EspnCfg:
    league_id: int
    team_id: int
    s2: str
    swid: str
    enabled: bool


@dataclass
class OddsCfg:
    provider: str
    api_key: str


@dataclass
class FantasyProsCfg:
    enabled: bool
    api_key: str
    base_url: str
    scoring_yahoo: str
    scoring_espn: str


@dataclass
class NtfyCfg:
    server: str
    topic: str
    priority: str
    token: str


@dataclass
class RemindersCfg:
    offsets_hours: list[float]
    quiet_start: int
    quiet_end: int


@dataclass
class Config:
    timezone: str
    dashboard_url: str
    supabase: Supabase
    yahoo: YahooCfg
    espn: EspnCfg
    odds: OddsCfg
    fantasypros: FantasyProsCfg
    ntfy: NtfyCfg
    reminders: RemindersCfg
    raw: dict = field(repr=False, default_factory=dict)


@lru_cache(maxsize=1)
def get_config() -> Config:
    load_dotenv(ROOT / ".env")

    yahoo_league = os.environ.get("YAHOO_LEAGUE_ID", "").strip()
    espn_league = os.environ.get("ESPN_LEAGUE_ID", "").strip()

    return Config(
        timezone=os.environ.get("TZ_NAME", "America/New_York"),
        dashboard_url=os.environ.get("DASHBOARD_URL", "https://squid004.github.io/nfl-hub/"),
        supabase=Supabase(
            url=os.environ.get("SUPABASE_URL", ""),
            anon_key=os.environ.get("SUPABASE_ANON_KEY", ""),
        ),
        yahoo=YahooCfg(
            league_id=yahoo_league,
            team_key=os.environ.get("YAHOO_TEAM_KEY", "").strip(),
            enabled=bool(yahoo_league) and OAUTH_PATH.exists(),
        ),
        espn=EspnCfg(
            league_id=int(espn_league) if espn_league.isdigit() else 0,
            team_id=int(os.environ.get("ESPN_TEAM_ID", "0") or 0),
            s2=os.environ.get("ESPN_S2", ""),
            swid=os.environ.get("ESPN_SWID", ""),
            enabled=espn_league.isdigit(),
        ),
        odds=OddsCfg(
            provider=os.environ.get("ODDS_PROVIDER", "espn").lower(),
            api_key=os.environ.get("ODDS_API_KEY", ""),
        ),
        fantasypros=FantasyProsCfg(
            enabled=_bool("FANTASYPROS_ENABLED", True) and bool(os.environ.get("FANTASYPROS_API_KEY", "")),
            api_key=os.environ.get("FANTASYPROS_API_KEY", ""),
            base_url=os.environ.get(
                "FANTASYPROS_BASE_URL", "https://api.fantasypros.com/public/v2/json"
            ).rstrip("/"),
            scoring_yahoo=os.environ.get("FP_SCORING_YAHOO", "PPR").upper(),
            scoring_espn=os.environ.get("FP_SCORING_ESPN", "PPR").upper(),
        ),
        ntfy=NtfyCfg(
            server=os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/"),
            topic=os.environ.get("NTFY_TOPIC", ""),
            priority=os.environ.get("NTFY_PRIORITY", "default"),
            token=os.environ.get("NTFY_TOKEN", ""),
        ),
        reminders=RemindersCfg(
            offsets_hours=_floats("REMINDER_OFFSETS", [24, 3, 0.75]),
            quiet_start=int(os.environ.get("QUIET_START", "23") or 23),
            quiet_end=int(os.environ.get("QUIET_END", "7") or 7),
        ),
    )
