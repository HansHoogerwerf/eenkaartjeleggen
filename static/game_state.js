/* ─── Klaverjassen Socket.IO client (multiplayer, i18n) ───────────────── */

const socket = io({
    reconnection: true,
    reconnectionAttempts: Infinity,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    // Connect-attempt timeout. A single hung attempt (bad proxy, captive
    // portal) must not block reconnection for longer than the server's
    // reconnect grace period.
    timeout: 10000,
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
   Stores {code, name, seat, token} in localStorage so the player is
   automatically reconnected on page reload. The token is the server-issued
   per-seat secret: with it the seat can be reclaimed even while the server
   still thinks an older socket holds it. The name is the fallback identity
   for a client that lost its token.
*/
const SESSION_KEY = "klaverjas_session";
function saveSession(code, name, extra = {}) {
    const prev = loadSession() || {};
    const next = {code, name};
    if (prev.code === code) {
        if (prev.token) next.token = prev.token;
        if (typeof prev.seat === "number") next.seat = prev.seat;
    }
    if (extra.token) next.token = extra.token;
    if (typeof extra.seat === "number") next.seat = extra.seat;
    try { localStorage.setItem(SESSION_KEY, JSON.stringify(next)); } catch(e) {}
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
     IDLE         — no session, sitting on the lobby form
     LOBBY        — in a room's waiting room (lobby visible, game not started)
     IN_GAME      — game running, this client has a confirmed seat
     RECONNECTING — socket dropped or game-state desync; trying to rejoin
     LEFT         — user explicitly left; do not auto-reconnect
     SUPERSEDED   — another tab/device took this seat over; stay dormant
                    until the user asks to take it back

   Transitions are triggered by Socket.IO events (room_created, room_joined,
   game_starting, game_state_snapshot, disconnect, session_superseded, etc.)
   and by user actions.
*/
const ConnState = Object.freeze({
    IDLE: "IDLE",
    LOBBY: "LOBBY",
    IN_GAME: "IN_GAME",
    RECONNECTING: "RECONNECTING",
    LEFT: "LEFT",
    SUPERSEDED: "SUPERSEDED",
});

const ConnectionFSM = {
    state: ConnState.IDLE,
    _rejoinTimer: null,      // pending retry of rejoin_game
    _snapshotTimer: null,    // watchdog: server must answer a rejoin
    _lastRejoinPayload: null,
    _rejoinAttempt: 0,
    _hiddenAt: null,
    _quiet: false,           // rejoin without flashing overlays (resync)

    // Back-off for "seat still held by an old socket": the server reaps a
    // dead socket within its ping timeout, so a few retries are enough.
    REJOIN_RETRY_DELAYS_MS: [2000, 4000, 8000],
    // A tab hidden for at least this long asks for a fresh snapshot when it
    // comes back, in case events were missed or the socket silently died.
    RESYNC_AFTER_HIDDEN_MS: 3000,
    // If a rejoin gets no answer, the socket is probably a zombie the
    // browser has not noticed yet: recycle it instead of waiting for the
    // ping timeout.
    SNAPSHOT_TIMEOUT_MS: 4000,

    set(next, opts = {}) {
        if (this.state === next) return;
        this.state = next;
        this._quiet = !!opts.quiet;
        this._applyOverlays();
    },

    is(state) { return this.state === state; },

    isDormant() {
        return this.state === ConnState.LEFT || this.state === ConnState.SUPERSEDED;
    },

    rejoinPayload() {
        const session = loadSession();
        if (!session || !session.code) return null;
        const payload = {code: session.code, name: session.name};
        if (session.token) payload.token = session.token;
        return payload;
    },

    /* Ask the server for our seat back, or for a fresh snapshot when we
       still hold it. Returns false when there is no session to rejoin. */
    rejoin(opts = {}) {
        const payload = this.rejoinPayload();
        if (!payload) return false;
        this.rejoinWith(payload, opts);
        return true;
    },

    /* Send a specific rejoin request (the stored session, or the name and
       code the user typed) and remember it for retries. */
    rejoinWith(payload, opts = {}) {
        this._lastRejoinPayload = payload;
        this.set(ConnState.RECONNECTING, {quiet: !!opts.quiet});
        socket.emit("rejoin_game", payload);
        this._armSnapshotTimer();
    },

    _armSnapshotTimer() {
        this._clearSnapshotTimer();
        this._snapshotTimer = setTimeout(() => {
            this._snapshotTimer = null;
            if (this.state !== ConnState.RECONNECTING) return;
            if (this._rejoinTimer) return;
            socket.disconnect();
            socket.connect();
        }, this.SNAPSHOT_TIMEOUT_MS);
    },

    _clearSnapshotTimer() {
        if (this._snapshotTimer) {
            clearTimeout(this._snapshotTimer);
            this._snapshotTimer = null;
        }
    },

    _clearRejoinTimer() {
        if (this._rejoinTimer) {
            clearTimeout(this._rejoinTimer);
            this._rejoinTimer = null;
        }
    },

    _serverAnswered() {
        this._clearSnapshotTimer();
        this._clearRejoinTimer();
        this._rejoinAttempt = 0;
    },

    /* Called by socket.on("connect"). If we have a session and aren't
       dormant, attempt to rejoin. */
    onSocketConnect() {
        if (this.isDormant()) return;
        this._rejoinAttempt = 0;
        this._clearRejoinTimer();
        // The server's reply (game_state_snapshot for an in-progress game,
        // room_joined for the lobby, or rejoin_error) moves us on.
        if (!this.rejoin()) this.set(ConnState.IDLE);
    },

    /* Called by socket.on("disconnect"). Show the paused overlay so the user
       gets immediate feedback while Socket.IO retries underneath. */
    onSocketDisconnect(reason) {
        this._clearRejoinTimer();
        this._clearSnapshotTimer();
        if (this.isDormant() || this.state === ConnState.IDLE) return;
        const wasInGame = this.state === ConnState.IN_GAME;
        this.set(ConnState.RECONNECTING);
        this._wasInGame = wasInGame;
        // The client library does not retry on its own after a disconnect
        // initiated by the server or by socket.disconnect().
        if (reason === "io server disconnect" || reason === "io client disconnect") {
            setTimeout(() => {
                if (this.state === ConnState.RECONNECTING && !socket.connected) socket.connect();
            }, 300);
        }
    },

    /* The seat is still held by an old socket and we have no token to prove
       it is ours. The server closes dead sockets within its ping timeout, so
       try again a few times before giving up. Returns true while retrying. */
    retryRejoinLater() {
        if (this._rejoinAttempt >= this.REJOIN_RETRY_DELAYS_MS.length) return false;
        const delay = this.REJOIN_RETRY_DELAYS_MS[this._rejoinAttempt++];
        this._clearRejoinTimer();
        this._clearSnapshotTimer();
        this._rejoinTimer = setTimeout(() => {
            this._rejoinTimer = null;
            if (this.state !== ConnState.RECONNECTING) return;
            const payload = this._lastRejoinPayload || this.rejoinPayload();
            if (!payload) return;
            if (socket.connected) {
                socket.emit("rejoin_game", payload);
                this._armSnapshotTimer();
            } else {
                socket.connect();
            }
        }, delay);
        return true;
    },

    /* Make sure we are live and in sync: reconnect a dead socket, or ask
       for a fresh snapshot over a live one (idempotent on the server). */
    resync() {
        if (this.isDormant() || this.state === ConnState.IDLE) return;
        if (!socket.connected) {
            socket.connect();
            return;
        }
        if (this.state === ConnState.RECONNECTING && (this._rejoinTimer || this._snapshotTimer)) return;
        this.rejoin({quiet: true});
    },

    onPageHidden() { this._hiddenAt = Date.now(); },

    onPageVisible() {
        const hiddenFor = this._hiddenAt ? Date.now() - this._hiddenAt : 0;
        this._hiddenAt = null;
        if (!socket.connected || hiddenFor >= this.RESYNC_AFTER_HIDDEN_MS) this.resync();
    },

    /* Called when the server confirms our seat (room_joined / room_created). */
    onLobbyJoined() {
        this._serverAnswered();
        this.set(ConnState.LOBBY);
    },

    /* Called when game_starting or game_state_snapshot arrives. */
    onGameJoined() {
        this._serverAnswered();
        this.set(ConnState.IN_GAME);
    },

    /* User explicitly left (leave_room or leave_game). No auto-reconnect. */
    onLeave() {
        this._serverAnswered();
        clearSession();
        this.set(ConnState.LEFT);
        // After a brief moment, allow new sessions
        setTimeout(() => {
            if (this.state === ConnState.LEFT) this.set(ConnState.IDLE);
        }, 300);
    },

    /* Another socket presented our token: go dormant, keep the (shared)
       session so the user can take the seat back from here. */
    onSuperseded() {
        this._serverAnswered();
        this.set(ConnState.SUPERSEDED);
    },

    /* User asked to take the seat back from a superseded tab. */
    onTakeOver() {
        this.state = ConnState.IDLE;   // leave the dormant state silently
        if (socket.connected) {
            this.rejoin();
        } else {
            this.set(ConnState.RECONNECTING);
            socket.connect();
        }
    },

    /* Hide all overlays and let the lobby take over. */
    onIdle() {
        this._serverAnswered();
        this.set(ConnState.IDLE);
    },

    _applyOverlays() {
        const paused = document.getElementById("paused-overlay");
        const lobbyOverlay = document.getElementById("lobby-overlay");
        if (!paused || !lobbyOverlay) return;

        if (this.state === ConnState.RECONNECTING) {
            if (this._quiet) return;
            const lobbyVisible = lobbyOverlay.classList.contains("active");
            if (lobbyVisible) {
                showAutoReconnecting();
            } else {
                const msgEl = document.getElementById("paused-msg");
                if (msgEl) msgEl.textContent = t("error.reconnecting");
                const hostActions = document.getElementById("paused-host-actions");
                if (hostActions) hostActions.style.display = "none";
                paused.classList.add("active");
            }
        } else if (this.state === ConnState.IN_GAME) {
            // Other seats may still be waiting to reconnect; the paused
            // overlay is rebuilt from the snapshot right after this.
            paused.classList.remove("active");
        } else {
            paused.classList.remove("active");
        }
    },
};

socket.on("connect", () => {
    ConnectionFSM.onSocketConnect();
});

socket.on("disconnect", (reason) => {
    ConnectionFSM.onSocketDisconnect(reason);
});

// A backgrounded tab (phone locked, app switched) often loses its socket
// without the browser noticing until the ping timeout. Resync as soon as
// the tab is back instead of waiting for that.
document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") ConnectionFSM.onPageHidden();
    else ConnectionFSM.onPageVisible();
});
window.addEventListener("online", () => ConnectionFSM.resync());
window.addEventListener("pageshow", (event) => {
    if (event.persisted) ConnectionFSM.resync();
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
