/* ─── Klaverjassen Socket.IO client (multiplayer, i18n) ───────────────── */

const socket = io({
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    timeout: 120000,
});

const RED_SUITS = new Set(["♦", "♥"]);
const SEAT_TEAMS = {0: 0, 1: 1, 2: 0, 3: 1};
const HAND_POS_CLASS = {0: "pos-s", 1: "pos-w", 2: "pos-n", 3: "pos-e"};

// Card sorting
const SUIT_ORDER = ["♣", "♥", "♠", "♦"];
const RANK_STRENGTH      = {"7":0,"8":1,"9":2,"J":3,"Q":4,"K":5,"10":6,"A":7};
const RANK_STRENGTH_TRUMP = {"7":0,"8":1,"Q":2,"K":3,"10":4,"A":5,"9":6,"J":7};

// Maps absolute seat → DOM element ids, rotated so mySeat is always at the bottom
const SEAT_CARD_IDS = ["south-cards", "west-cards", "north-cards", "east-cards"];
const SEAT_LABEL_IDS = ["label-south", "label-west", "label-north", "label-east"];
// Visual positions: 0=bottom, 1=left, 2=top, 3=right
const VISUAL_TRICK_SLOTS = ["trick-s", "trick-w", "trick-n", "trick-e"];

let mySeat = 0;
let roomCode = null;
let isCreator = false;            // alias of isHost (kept for legacy callers)
let isHost = false;
let playerNames = {};
let currentLegal = [];
let teamNames = ["Team 0", "Team 1"];
let currentTrump = null;        // track trump for strength-aware default sort
let userHandOrder = [];         // card strings in user's preferred order (drag-to-reorder)
let dragSrcIndex = null;        // index of card being dragged
let trickPlayCount = 0;         // cards played so far in the current trick (0–4)

/* ─── Session persistence ────────────────────────────────────────────────
   Stores {code, name} in localStorage so the player is automatically
   reconnected on page reload. Name is the stable identity for reconnect.
*/
const SESSION_KEY = "klaverjas_session";
function saveSession(code, name) {
    try { localStorage.setItem(SESSION_KEY, JSON.stringify({code, name})); } catch(e) {}
    try { localStorage.setItem("klaverjas_player_name", name); } catch(e) {}
}
function loadSession() {
    try { const s = localStorage.getItem(SESSION_KEY); return s ? JSON.parse(s) : null; } catch(e) { return null; }
}
function clearSession() {
    try { localStorage.removeItem(SESSION_KEY); } catch(e) {}
}
function loadPlayerName() {
    try { return localStorage.getItem("klaverjas_player_name") || null; } catch(e) { return null; }
}

/* ─── ConnectionFSM ──────────────────────────────────────────────────────
   Single source of truth for the client connection state. The UI overlays
   and rejoin attempts are driven from this — no scattered ad-hoc handlers.

   States:
     IDLE       — no session, sitting on the lobby form
     LOBBY      — in a room's waiting room (lobby visible, game not started)
     IN_GAME    — game running, this client has a confirmed seat
     RECONNECTING — socket dropped or game-state desync; trying to rejoin
     LEFT       — user explicitly left; do not auto-reconnect

   Transitions are triggered by Socket.IO events (room_created, room_joined,
   game_starting, game_state_snapshot, disconnect, etc.) and by user actions.
*/
const ConnState = Object.freeze({
    IDLE: "IDLE",
    LOBBY: "LOBBY",
    IN_GAME: "IN_GAME",
    RECONNECTING: "RECONNECTING",
    LEFT: "LEFT",
});

const ConnectionFSM = {
    state: ConnState.IDLE,
    _rejoinTimer: null,

    set(next) {
        if (this.state === next) return;
        this.state = next;
        this._applyOverlays();
    },

    is(state) { return this.state === state; },

    /* Called by socket.on("connect"). If we have a session and aren't in the
       LEFT state, attempt to rejoin. */
    onSocketConnect() {
        if (this.state === ConnState.LEFT) return;
        const session = loadSession();
        if (!session) {
            this.set(ConnState.IDLE);
            return;
        }
        // We have a session — rejoin. The server's reply (game_state_snapshot
        // for an in-progress game, room_joined for the lobby, or rejoin_error)
        // will move us to the right state.
        this.set(ConnState.RECONNECTING);
        socket.emit("rejoin_game", {code: session.code, name: session.name});
    },

    /* Called by socket.on("disconnect"). Show the paused overlay so the user
       gets immediate feedback while Socket.IO retries underneath. */
    onSocketDisconnect() {
        if (this.state === ConnState.LEFT || this.state === ConnState.IDLE) return;
        // Stay logically in the same room state, but switch to RECONNECTING
        // overlay treatment.
        const wasInGame = this.state === ConnState.IN_GAME;
        this.set(ConnState.RECONNECTING);
        // Remember whether to return to game or lobby on success
        this._wasInGame = wasInGame;
    },

    /* Called when the server confirms our seat (room_joined / room_created). */
    onLobbyJoined() {
        this.set(ConnState.LOBBY);
    },

    /* Called when game_starting or game_state_snapshot arrives. */
    onGameJoined() {
        this.set(ConnState.IN_GAME);
    },

    /* User explicitly left (leave_room or leave_game). No auto-reconnect. */
    onLeave() {
        clearSession();
        this.set(ConnState.LEFT);
        // After a brief moment, allow new sessions
        setTimeout(() => {
            if (this.state === ConnState.LEFT) this.set(ConnState.IDLE);
        }, 300);
    },

    /* Hide all overlays and let the lobby take over. */
    onIdle() {
        this.set(ConnState.IDLE);
    },

    _applyOverlays() {
        const paused = document.getElementById("paused-overlay");
        const lobbyOverlay = document.getElementById("lobby-overlay");
        if (!paused || !lobbyOverlay) return;

        if (this.state === ConnState.RECONNECTING) {
            const lobbyVisible = lobbyOverlay.classList.contains("active");
            if (lobbyVisible) {
                showAutoReconnecting();
            } else {
                const msgEl = document.getElementById("paused-msg");
                if (msgEl) msgEl.textContent = t("error.reconnecting");
                paused.classList.add("active");
            }
        } else if (this.state === ConnState.IN_GAME) {
            paused.classList.remove("active");
        } else if (this.state === ConnState.LOBBY || this.state === ConnState.IDLE) {
            paused.classList.remove("active");
        }
    },
};

socket.on("connect", () => {
    ConnectionFSM.onSocketConnect();
});

socket.on("disconnect", () => {
    ConnectionFSM.onSocketDisconnect();
});

/* ─── Seat rotation ───────────────────────────────────────────────────
   Maps an absolute seat index to a visual position (0=bottom, 1=left, 2=top, 3=right)
   so that mySeat always appears at the bottom of the screen.
*/
function visualPos(absSeat) {
    return (absSeat - mySeat + 4) % 4;
}

function cardContainerId(absSeat) {
    return SEAT_CARD_IDS[visualPos(absSeat)];
}

function labelId(absSeat) {
    return SEAT_LABEL_IDS[visualPos(absSeat)];
}

function trickSlotId(absSeat) {
    return VISUAL_TRICK_SLOTS[visualPos(absSeat)];
}

/* ─── Helpers ─────────────────────────────────────────────────────────── */

function cardStr(c) { return c.rank + c.suit; }
function isRed(suit) { return RED_SUITS.has(suit); }

function makeCardFaceHTML(card, extra = "") {
    const color = isRed(card.suit) ? "red" : "";
    return `<div class="card card-face ${color} ${extra}" data-card="${cardStr(card)}">
        <span>${card.rank}</span><span>${card.suit}</span>
    </div>`;
}

function makeCardBackSmallHTML() {
    return `<div class="card-back-small"></div>`;
}

/* ─── Rendering ───────────────────────────────────────────────────────── */

