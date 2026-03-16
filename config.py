"""Centralized runtime configuration for server and room/game defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerConfig:
    secret_key: str
    cors_origins: str
    ping_timeout: int
    ping_interval: int
    host: str
    port: int
    debug: bool


@dataclass(frozen=True)
class RoomConfig:
    default_player_name: str
    max_player_name_len: int
    max_chat_message_len: int
    seat_count: int
    room_code_length: int
    default_game_mode: str
    allowed_game_modes: tuple[str, ...]
    default_ai_strength: str
    allowed_ai_strengths: tuple[str, ...]
    default_rules_variant: str
    allowed_rules_variants: tuple[str, ...]
    default_score_limit: int
    min_score_limit: int
    max_score_limit: int
    default_team_names: tuple[str, str]
    max_team_name_len: int


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig
    room: RoomConfig


CONFIG = AppConfig(
    server=ServerConfig(
        secret_key=os.environ.get("SECRET_KEY", "klaverjas-secret"),
        cors_origins=os.environ.get("CORS_ORIGINS", "*"),
        ping_timeout=30,
        ping_interval=10,
        host="0.0.0.0",
        port=5000,
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    ),
    room=RoomConfig(
        default_player_name="Player",
        max_player_name_len=16,
        max_chat_message_len=200,
        seat_count=4,
        room_code_length=4,
        default_game_mode="score_limit",
        allowed_game_modes=("score_limit", "boom", "free_play"),
        default_ai_strength="expert",
        allowed_ai_strengths=("beginner", "advanced", "expert", "expert_v2"),
        default_rules_variant="rotterdam",
        allowed_rules_variants=("rotterdam", "amsterdam"),
        default_score_limit=500,
        min_score_limit=50,
        max_score_limit=5000,
        default_team_names=("Team A", "Team N"),
        max_team_name_len=16,
    ),
)
