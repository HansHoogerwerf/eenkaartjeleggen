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
let isCreator = false;
let playerNames = {};
let currentLegal = [];
let teamNames = ["Team 0", "Team 1"];
let currentTrump = null;        // track trump for strength-aware default sort
let userHandOrder = [];         // card strings in user's preferred order (drag-to-reorder)
let dragSrcIndex = null;        // index of card being dragged
let trickPlayCount = 0;         // cards played so far in the current trick (0–4)

/* ─── Session persistence (auto-reconnect on page reload) ─────────────────
   Stores {code, name} in localStorage so the player is automatically
   put back into their seat if they reload or briefly lose connection.
   Cleared when the game ends normally.
*/
const SESSION_KEY = "klaverjas_session";
function saveSession(code, name) {
    try { localStorage.setItem(SESSION_KEY, JSON.stringify({code, name})); } catch(e) {}
}
function loadSession() {
    try { const s = localStorage.getItem(SESSION_KEY); return s ? JSON.parse(s) : null; } catch(e) { return null; }
}
function clearSession() {
    try { localStorage.removeItem(SESSION_KEY); } catch(e) {}
}

/* On every socket connection (initial load AND mid-game reconnects):
   - If the lobby is visible and we have a session → show reconnecting UI
   - If the game is already showing (mid-game socket reconnect) → silently rejoin
     and hide the paused overlay once the server confirms with "reconnected".
*/
socket.on("connect", () => {
    const session = loadSession();
    if (!session) return;
    const lobbyVisible = document.getElementById("lobby-overlay").classList.contains("active");
    if (lobbyVisible) {
        showAutoReconnecting();
    }
    socket.emit("join_room", {code: session.code, name: session.name});
});

/* Show the paused overlay immediately when our own socket drops, so the
   player sees feedback at once rather than waiting for the server ping
   timeout to fire (which can take 10–40 s). */
socket.on("disconnect", () => {
    const lobbyVisible = document.getElementById("lobby-overlay").classList.contains("active");
    if (!lobbyVisible) {
        document.getElementById("paused-msg").textContent = t("error.reconnecting");
        document.getElementById("paused-overlay").classList.add("active");
    }
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

