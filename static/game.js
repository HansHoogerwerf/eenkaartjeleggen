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

function updateSeatLabels() {
    for (let seat = 0; seat < 4; seat++) {
        const el = document.getElementById(labelId(seat));
        const name = playerNames[seat] || tSeat(seat);
        const suffix = seat === mySeat ? t("seat.you_suffix") : "";
        el.textContent = name + suffix;
    }
}

/* Compare two card objects for default (non-user) sort:
   group by SUIT_ORDER, then strongest first within each suit. */
function defaultCardCompare(a, b) {
    const si = SUIT_ORDER.indexOf(a.suit), sj = SUIT_ORDER.indexOf(b.suit);
    if (si !== sj) return si - sj;
    const table = s => s === currentTrump ? RANK_STRENGTH_TRUMP : RANK_STRENGTH;
    return (table(b.suit)[b.rank] ?? 0) - (table(a.suit)[a.rank] ?? 0); // strongest first
}

function renderMyHand(cards, legal) {
    const el = document.getElementById(cardContainerId(mySeat));
    const legalSet = new Set(legal || []);

    // Sync userHandOrder: drop cards no longer in hand
    const cardSet = new Set(cards.map(cardStr));
    userHandOrder = userHandOrder.filter(cs => cardSet.has(cs));

    // Any cards not yet in userHandOrder (fresh deal) → default-sort them and append
    const tracked = new Set(userHandOrder);
    const newCards = cards.filter(c => !tracked.has(cardStr(c)));
    if (newCards.length > 0) {
        newCards.sort(defaultCardCompare);
        userHandOrder.push(...newCards.map(cardStr));
    }

    // Render in userHandOrder sequence
    const orderMap = new Map(userHandOrder.map((cs, i) => [cs, i]));
    const sorted = [...cards].sort((a, b) =>
        (orderMap.get(cardStr(a)) ?? 999) - (orderMap.get(cardStr(b)) ?? 999));

    el.innerHTML = sorted.map(c => {
        const cs = cardStr(c);
        const cls = legalSet.has(cs) ? "legal" : "disabled";
        return makeCardFaceHTML(c, cls);
    }).join("");

    el.querySelectorAll(".card-face.legal").forEach(div => {
        div.addEventListener("click", () => {
            socket.emit("play_card", {card: div.dataset.card});
            el.querySelectorAll(".card-face.legal").forEach(d => {
                d.classList.remove("legal");
                d.classList.add("disabled");
            });
        });
    });

    initDragAndDrop(el);
}

/* Allow the player to drag cards within their hand to reorder them. */
function initDragAndDrop(el) {
    el.querySelectorAll(".card-face").forEach((card, idx) => {
        card.setAttribute("draggable", "true");

        card.addEventListener("dragstart", e => {
            dragSrcIndex = idx;
            e.dataTransfer.effectAllowed = "move";
            // Defer class addition so ghost image captures the normal card appearance
            requestAnimationFrame(() => card.classList.add("dragging"));
        });

        card.addEventListener("dragend", () => {
            card.classList.remove("dragging");
            el.querySelectorAll(".card-face").forEach(c => c.classList.remove("drag-over"));
            dragSrcIndex = null;
        });

        card.addEventListener("dragover", e => {
            e.preventDefault();
            e.dataTransfer.dropEffect = "move";
            el.querySelectorAll(".card-face").forEach(c => c.classList.remove("drag-over"));
            if (idx !== dragSrcIndex) card.classList.add("drag-over");
        });

        card.addEventListener("drop", e => {
            e.preventDefault();
            if (dragSrcIndex === null || dragSrcIndex === idx) return;

            // Reorder userHandOrder
            const newOrder = [...userHandOrder];
            const [moved] = newOrder.splice(dragSrcIndex, 1);
            newOrder.splice(idx, 0, moved);
            userHandOrder = newOrder;
            dragSrcIndex = null;

            // Re-render with preserved legal state
            const allCards = [...el.querySelectorAll(".card-face")].map(div => {
                const cs = div.dataset.card;
                return {rank: cs.slice(0, -1), suit: cs.slice(-1)};
            });
            renderMyHand(allCards, [...currentLegal]);
        });
    });
}

function renderOtherCards(cardCounts) {
    for (const [seatStr, count] of Object.entries(cardCounts)) {
        const seat = parseInt(seatStr);
        if (seat === mySeat) continue;
        const el = document.getElementById(cardContainerId(seat));
        el.innerHTML = Array(count).fill(makeCardBackSmallHTML()).join("");
    }
}

function showCardInTrick(playerIdx, card) {
    const slot = document.getElementById(trickSlotId(playerIdx));
    const animClass = ["anim-s", "anim-w", "anim-n", "anim-e"][visualPos(playerIdx)];
    slot.className = `trick-slot filled ${animClass}`;
    slot.style.color = isRed(card.suit) ? "var(--red-suit)" : "var(--black-suit)";
    slot.innerHTML = `<span>${card.rank}</span><span>${card.suit}</span>`;
    slot.addEventListener("animationend", () => slot.classList.remove(animClass), {once: true});
}

function showTurnArrow(absSeat) {
    const arrow = document.getElementById("trick-arrow");
    if (!arrow) return;
    const dirClass = ["dir-s", "dir-w", "dir-n", "dir-e"][visualPos(absSeat)];
    arrow.className = `visible ${dirClass}`;
}

function hideTurnArrow() {
    const arrow = document.getElementById("trick-arrow");
    if (arrow) arrow.className = "";
}

function showBidBadge(absSeat, declared) {
    const labelEl = document.getElementById(labelId(absSeat));
    // Remove any existing badge first
    labelEl.querySelectorAll(".bid-badge").forEach(b => b.remove());
    const badge = document.createElement("span");
    badge.className = `bid-badge ${declared ? "bid-declared" : "bid-passed"}`;
    badge.textContent = declared ? t("bid.badge_declared") : t("bid.badge_passed");
    labelEl.appendChild(badge);
    setTimeout(() => badge.remove(), 2600);
}

function clearTrickArea() {
    for (const id of VISUAL_TRICK_SLOTS) {
        const slot = document.getElementById(id);
        slot.className = "trick-slot empty";
        slot.innerHTML = "";
        slot.style.color = "";
    }
}

function clearDeclaringHighlight() {
    for (const id of SEAT_LABEL_IDS) {
        document.getElementById(id).classList.remove("declaring");
    }
}

function updateScores(scores) {
    document.getElementById("score-t0").textContent = t("score.team0", {name: teamNames[0], score: scores[0]});
    document.getElementById("score-t1").textContent = t("score.team1", {name: teamNames[1], score: scores[1]});
}

function updateRoundScores(tricks, roem) {
    const t0 = tricks[0], t1 = tricks[1], r0 = roem[0], r1 = roem[1];
    document.getElementById("round-t0").textContent =
        t("score.round_team0", {name: teamNames[0], roem: r0, tricks: t0, total: r0 + t0});
    document.getElementById("round-t1").textContent =
        t("score.round_team1", {name: teamNames[1], roem: r1, tricks: t1, total: r1 + t1});
}

function appendLog(msg, tag) {
    // Log panel removed — stub kept for server compatibility
}

/* ─── Lobby ───────────────────────────────────────────────────────────── */

function createRoom() {
    const name = document.getElementById("lobby-name").value.trim() || t("lobby.name_placeholder");
    socket.emit("create_room", {name});
}

function peekRoom() {
    const code = document.getElementById("join-code").value.trim().toUpperCase();
    if (!code || code.length < 4) {
        showLobbyError(t("error.enter_code"));
        return;
    }
    socket.emit("peek_room", {code});
}

function joinRoom(seat) {
    const name = document.getElementById("lobby-name").value.trim() || t("lobby.name_placeholder");
    const code = document.getElementById("join-code").value.trim().toUpperCase();
    socket.emit("join_room", {code, name, seat});
}

function showSeatPicker(lobby) {
    document.getElementById("lobby-reconnecting").style.display = "none";
    document.getElementById("lobby-actions").style.display = "none";
    document.getElementById("lobby-name-section").style.display = "none";
    document.getElementById("lobby-seat-picker").style.display = "block";

    const grid = document.getElementById("seat-picker-grid");
    grid.innerHTML = "";
    for (let i = 0; i < 4; i++) {
        const seat = lobby.seats[String(i)];
        const teamLabel = t("team." + seat.team);
        const taken = seat.is_human;
        const playerName = taken ? seat.name : t("seat.open");
        const cls = taken ? "taken" : "open";
        const row = document.createElement("div");
        row.className = `lobby-seat-row seat-pick-row ${cls}`;
        row.innerHTML = `<span class="seat-num">${tSeat(i)}</span>
            <span class="seat-player ${taken ? '' : 'empty'}">${playerName}</span>
            <span class="seat-team">${teamLabel}</span>`;
        if (!taken) {
            row.style.cursor = "pointer";
            row.addEventListener("click", () => joinRoom(i));
        }
        grid.appendChild(row);
    }
}

function cancelSeatPicker() {
    document.getElementById("lobby-seat-picker").style.display = "none";
    document.getElementById("lobby-actions").style.display = "block";
    document.getElementById("lobby-name-section").style.display = "block";
}

function showAutoReconnecting() {
    document.getElementById("lobby-reconnecting").style.display = "block";
    document.getElementById("lobby-name-section").style.display = "none";
    document.getElementById("lobby-actions").style.display = "none";
}

function cancelAutoReconnect() {
    clearSession();
    document.getElementById("lobby-reconnecting").style.display = "none";
    document.getElementById("lobby-name-section").style.display = "block";
    document.getElementById("lobby-actions").style.display = "block";
}

let selectedMode = "score_limit";

function selectMode(mode) {
    selectedMode = mode;
    document.querySelectorAll(".mode-btn").forEach(btn => {
        if (btn.dataset.mode === mode) {
            btn.className = "btn-declare mode-btn active";
        } else {
            btn.className = "btn-pass mode-btn";
        }
    });
    document.getElementById("score-limit-input").style.display =
        mode === "score_limit" ? "block" : "none";
}

function startGame() {
    const data = { mode: selectedMode };
    if (selectedMode === "score_limit") {
        const el = document.getElementById("mode-score-limit");
        data.score_limit = el ? (parseInt(el.value) || 500) : 500;
    }
    const t0 = document.getElementById("team0-name");
    const t1 = document.getElementById("team1-name");
    data.team_names = [
        t0 ? (t0.value.trim() || "Team 0") : "Team 0",
        t1 ? (t1.value.trim() || "Team 1") : "Team 1",
    ];
    socket.emit("start_game", data);
}

function showLobbyError(msg) {
    const el = document.getElementById("lobby-error");
    el.textContent = msg;
    el.style.display = "block";
    setTimeout(() => el.style.display = "none", 4000);
}

function showWaitingRoom(lobby) {
    document.getElementById("lobby-reconnecting").style.display = "none";
    document.getElementById("lobby-actions").style.display = "none";
    document.getElementById("lobby-name-section").style.display = "none";
    document.getElementById("lobby-waiting").style.display = "block";
    document.getElementById("lobby-code").textContent = lobby.code;
    updateLobbySeats(lobby);
}

function updateLobbySeats(lobby) {
    const container = document.getElementById("lobby-seats");
    container.innerHTML = "";
    for (let i = 0; i < 4; i++) {
        const seat = lobby.seats[String(i)];
        const teamLabel = t("team." + seat.team);
        const playerName = seat.is_human ? seat.name : t("seat.ai");
        const cls = seat.is_human ? "" : "empty";
        container.innerHTML += `<div class="lobby-seat-row">
            <span class="seat-num">${tSeat(i)}</span>
            <span class="seat-player ${cls}">${playerName}</span>
            <span class="seat-team">${teamLabel}</span>
        </div>`;
    }

    // Show start button and mode picker only for creator
    document.getElementById("lobby-start-btn").style.display = isCreator ? "inline-block" : "none";
    document.getElementById("lobby-wait-msg").style.display = isCreator ? "none" : "block";

    const teamNamesEl = document.getElementById("lobby-team-names");
    if (teamNamesEl) {
        teamNamesEl.style.display = isCreator ? "block" : "none";
    }

    const modePicker = document.getElementById("lobby-mode-picker");
    const modeDisplay = document.getElementById("lobby-mode-display");
    if (modePicker) {
        modePicker.style.display = isCreator ? "block" : "none";
    }
    if (modeDisplay) {
        if (!isCreator) {
            const modeText = t("mode." + selectedMode);
            modeDisplay.style.display = "block";
            document.getElementById("lobby-mode-text").textContent = modeText;
        } else {
            modeDisplay.style.display = "none";
        }
    }
}

// Lobby socket events

socket.on("room_created", data => {
    roomCode = data.code;
    mySeat = data.seat;
    isCreator = true;
    const name = document.getElementById("lobby-name").value.trim()
        || (loadSession() || {}).name
        || t("lobby.name_placeholder");
    saveSession(data.code, name);
    showWaitingRoom(data.lobby);
});

socket.on("room_joined", data => {
    roomCode = data.code;
    mySeat = data.seat;
    isCreator = false;
    const name = document.getElementById("lobby-name").value.trim()
        || (loadSession() || {}).name
        || t("lobby.name_placeholder");
    saveSession(data.code, name);
    showWaitingRoom(data.lobby);
});

socket.on("room_peeked", data => {
    showSeatPicker(data.lobby);
});

socket.on("join_error", data => {
    if (document.getElementById("lobby-reconnecting").style.display !== "none") {
        // Auto-reconnect failed — pre-fill form so user can try manually
        const session = loadSession();
        if (session) {
            const nameEl = document.getElementById("lobby-name");
            const codeEl = document.getElementById("join-code");
            if (nameEl && !nameEl.value) nameEl.value = session.name;
            if (codeEl && !codeEl.value) codeEl.value = session.code;
        }
        cancelAutoReconnect(); // also clears session
    }
    const msg = data.key ? t(data.key) : (data.msg || "Error");
    showLobbyError(msg);
});

socket.on("lobby_update", data => {
    if (document.getElementById("lobby-waiting").style.display !== "none") {
        updateLobbySeats(data);
    }
});

socket.on("game_starting", data => {
    mySeat = data.seat;
    playerNames = data.player_names;
    if (data.team_names) teamNames = data.team_names;

    // Hide lobby, show game
    document.getElementById("lobby-overlay").classList.remove("active");
    updateSeatLabels();
    updateScores([0, 0]);

    // Show New Game button only for creator; Leave button for everyone
    document.getElementById("new-game-btn").style.display = isCreator ? "inline-block" : "none";
    document.getElementById("gameover-newgame-btn").style.display = isCreator ? "inline-block" : "none";
    document.getElementById("leave-game-btn").style.display = "inline-block";
});

/* ─── Game socket events ──────────────────────────────────────────────── */

socket.on("deal_done", data => {
    document.getElementById("nextround-banner").classList.remove("active");
    clearDeclaringHighlight();
    const trumpInd = document.getElementById("trump-card-indicator");
    if (trumpInd) trumpInd.classList.remove("active");
    clearTrickArea();
    if (data.my_seat !== undefined) mySeat = data.my_seat;
    // Reset hand state for the new round
    currentTrump = null;
    userHandOrder = [];
    hideTurnArrow();
    trickPlayCount = 0;
    renderMyHand(data.hand, []);
    if (data.card_counts) renderOtherCards(data.card_counts);
    updateRoundScores([0, 0], [0, 0]);
});

socket.on("trump_offered", data => {
    const color = isRed(data.suit) ? "red" : "var(--accent)";
    document.getElementById("trump-label").innerHTML =
        `<span style="color:${color}">${t("score.offered", {suit: data.suit, name: tSuit(data.suit), round: data.round_num})}</span>`;
});

socket.on("bid", data => {
    showBidBadge(data.player_idx, data.trump !== null);
});

socket.on("trump_set", data => {
    document.getElementById("trump-label").innerHTML =
        `${t("score.trump", {suit: data.trump, team: data.declaring_team, player: data.declaring_player})}`;

    // Highlight declaring player's seat label
    clearDeclaringHighlight();
    if (data.declaring_player_idx !== undefined) {
        document.getElementById(labelId(data.declaring_player_idx)).classList.add("declaring");
    }

    // Show trump card on table
    const indicator = document.getElementById("trump-card-indicator");
    const card = document.getElementById("trump-card");
    if (indicator && card) {
        card.className = isRed(data.trump) ? "card red" : "card";
        card.innerHTML = `<span>${data.trump}</span><span>${data.trump}</span>`;
        indicator.classList.add("active");
    }

    currentTrump = data.trump;
    userHandOrder = [];           // re-sort with trump-aware ordering
    if (data.hand) renderMyHand(data.hand, []);

    // Show arrow pointing to the first trick leader
    if (data.leader_idx !== undefined) {
        trickPlayCount = 0;
        showTurnArrow(data.leader_idx);
    }
});

socket.on("trick_played", data => {
    trickPlayCount++;
    if (trickPlayCount < 4) {
        showTurnArrow((data.player_idx + 1) % 4);
    } else {
        hideTurnArrow(); // all 4 played, waiting for trick_cleared
    }
    showCardInTrick(data.player_idx, data.card);
    if (data.player_idx === mySeat) currentLegal = [];
    if (data.hand) renderMyHand(data.hand, currentLegal);
    if (data.card_counts) renderOtherCards(data.card_counts);
});

socket.on("trick_won", data => {
    if (data.cur_tricks && data.cur_roem)
        updateRoundScores(data.cur_tricks, data.cur_roem);
});

socket.on("trick_cleared", data => {
    clearTrickArea();
    currentLegal = [];
    trickPlayCount = 0;
    if (data && data.next_leader !== undefined) showTurnArrow(data.next_leader);
});

let lastRoundResult = null;

socket.on("round_done", data => {
    updateScores(data.scores);
    updateRoundScores([0, 0], [0, 0]);
    lastRoundResult = data;
});

socket.on("roem", () => {});
socket.on("nat", () => {});

socket.on("waiting_for_host", () => {
    const banner = document.getElementById("nextround-banner");
    document.getElementById("nextround-btn").style.display = isCreator ? "inline-block" : "none";
    document.getElementById("nextround-wait").style.display = isCreator ? "none" : "inline";

    // Show round outcome
    const msg = document.getElementById("nextround-msg");
    if (lastRoundResult) {
        msg.textContent = `${teamNames[0]}: +${lastRoundResult.t0}   |   ${teamNames[1]}: +${lastRoundResult.t1}`;
    }

    banner.classList.add("active");
});

socket.on("game_over", data => {
    clearSession();
    const s = data.scores;
    document.getElementById("gameover-msg").textContent =
        t("modal.game_over_msg", {winner: teamNames[data.winner], team0: teamNames[0], team1: teamNames[1], s0: s[0], s1: s[1]});
    document.getElementById("gameover-overlay").classList.add("active");
});

socket.on("log", data => {
    appendLog(data.msg, data.tag);
});

/* ─── Human input requests ────────────────────────────────────────────── */

socket.on("request_move", data => {
    currentLegal = data.legal;
    const cards = [];
    document.querySelectorAll(`#${cardContainerId(mySeat)} .card-face`).forEach(div => {
        const cs = div.dataset.card;
        const suit = cs.slice(-1);
        const rank = cs.slice(0, -1);
        cards.push({rank, suit});
    });
    renderMyHand(cards, data.legal);
});

socket.on("request_bid", data => {
    const overlay = document.getElementById("bid-overlay");
    const suitDiv = document.getElementById("bid-suit");
    const msgDiv = document.getElementById("bid-msg");
    const btnsDiv = document.getElementById("bid-buttons");

    suitDiv.className = `bid-suit ${isRed(data.suit) ? "red" : "black"}`;
    suitDiv.textContent = `${data.suit}  ${tSuit(data.suit)}`;

    msgDiv.textContent = data.forced
        ? t("bid.forced")
        : t("bid.optional");

    btnsDiv.innerHTML = "";

    const declareBtn = document.createElement("button");
    declareBtn.className = "btn-declare";
    declareBtn.textContent = t("bid.declare");
    declareBtn.onclick = () => {
        socket.emit("bid_response", {declare: true});
        overlay.classList.remove("active");
    };
    btnsDiv.appendChild(declareBtn);

    if (!data.forced) {
        const passBtn = document.createElement("button");
        passBtn.className = "btn-pass";
        passBtn.textContent = t("bid.pass");
        passBtn.onclick = () => {
            socket.emit("bid_response", {declare: false});
            overlay.classList.remove("active");
        };
        btnsDiv.appendChild(passBtn);
    }

    overlay.classList.add("active");
});

/* ─── Disconnect / Reconnect ──────────────────────────────────────────── */

socket.on("player_disconnected", data => {
    document.getElementById("paused-msg").textContent = t("error.waiting_reconnect", {name: data.name});
    document.getElementById("paused-overlay").classList.add("active");
});

socket.on("game_paused", data => {
    const msg = data.key ? t(data.key, {name: data.name}) : data.msg;
    document.getElementById("paused-msg").textContent = msg;
    document.getElementById("paused-overlay").classList.add("active");
});

socket.on("player_reconnected", data => {
    document.getElementById("paused-overlay").classList.remove("active");
    appendLog(t("log.reconnected", {name: data.name}), "winner");
});

socket.on("reconnected", data => {
    // We reconnected to a running game
    roomCode = data.code;
    mySeat = data.seat;
    playerNames = data.player_names;
    if (data.is_creator !== undefined) isCreator = data.is_creator;
    if (data.team_names) teamNames = data.team_names;

    document.getElementById("lobby-overlay").classList.remove("active");
    document.getElementById("paused-overlay").classList.remove("active");
    document.getElementById("leave-game-btn").style.display = "inline-block";
    document.getElementById("new-game-btn").style.display = isCreator ? "inline-block" : "none";
    document.getElementById("gameover-newgame-btn").style.display = isCreator ? "inline-block" : "none";
    updateSeatLabels();
    updateScores(data.scores);
    updateRoundScores(data.cur_tricks, data.cur_roem);

    if (data.hand) renderMyHand(data.hand, []);
    if (data.card_counts) renderOtherCards(data.card_counts);

    // Restore trump indicator
    if (data.trump) {
        document.getElementById("trump-label").innerHTML =
            t("score.trump", {suit: data.trump, team: data.declaring_team, player: data.declaring_player});
        clearDeclaringHighlight();
        if (data.declaring_player_idx !== null && data.declaring_player_idx !== undefined) {
            document.getElementById(labelId(data.declaring_player_idx)).classList.add("declaring");
        }
        const indicator = document.getElementById("trump-card-indicator");
        const card = document.getElementById("trump-card");
        if (indicator && card) {
            card.className = isRed(data.trump) ? "card red" : "card";
            card.innerHTML = `<span>${data.trump}</span><span>${data.trump}</span>`;
            indicator.classList.add("active");
        }
    }

    // Restore any cards already played in the current trick
    if (data.trick_cards) {
        for (const [pidxStr, c] of Object.entries(data.trick_cards)) {
            showCardInTrick(parseInt(pidxStr), c);
        }
    }

});

/* ─── Hands viewer ────────────────────────────────────────────────────── */

socket.on("hands_data", data => {
    const container = document.getElementById("hands-content");
    const tricks = data.tricks;
    const players = data.players;

    if (!tricks || tricks.length === 0) {
        container.innerHTML = `<p style='text-align:center;color:#aaa;'>${t("hands.empty")}</p>`;
        document.getElementById("hands-overlay").classList.add("active");
        return;
    }

    container.innerHTML = tricks.map((trick, i) => {
        const header = t("hands.trick_header", {num: i + 1, winner: trick.winner_name, pts: trick.pts});

        const cardCells = [0, 1, 2, 3].map(pidx => {
            const card = trick.cards[pidx];
            if (!card) return "";
            const isWinner = pidx === trick.winner_idx;
            const colorCls = isRed(card.suit) ? "red" : "black";
            const winCls = isWinner ? "winner" : "";
            const posClass = HAND_POS_CLASS[pidx];
            return `<div class="hand-card ${colorCls} ${winCls} ${posClass}">
                <div class="card-name">${players[pidx] || "?"}</div>
                <div class="card-val">${card.rank}${card.suit}</div>
            </div>`;
        }).join("");

        return `<div class="hand-trick">
            <div class="hand-trick-header">${header}</div>
            <div class="hand-trick-grid">${cardCells}</div>
        </div>`;
    }).join("");

    document.getElementById("hands-overlay").classList.add("active");
});

function openHands() {
    socket.emit("get_hands");
}

/* ─── History viewer ──────────────────────────────────────────────────── */

socket.on("history_data", data => {
    const rounds = data.rounds;
    const listEl = document.getElementById("history-list");
    const detailEl = document.getElementById("history-detail");

    if (!rounds || rounds.length === 0) {
        listEl.innerHTML = `<p style='color:#aaa;'>${t("history.empty")}</p>`;
        detailEl.textContent = "";
        document.getElementById("history-overlay").classList.add("active");
        return;
    }

    listEl.innerHTML = rounds.map((r, i) => {
        const nat = r.nat ? t("history.nat_short") : "";
        return `<div class="history-list-item" data-idx="${i}">
            ${t("history.round_summary", {num: String(r.round_num).padStart(2), t0: r.round_pts[0], t1: r.round_pts[1], nat})}
        </div>`;
    }).join("");

    function showRound(idx) {
        const r = rounds[idx];
        listEl.querySelectorAll(".history-list-item").forEach(el => el.classList.remove("selected"));
        listEl.querySelector(`[data-idx="${idx}"]`).classList.add("selected");

        let text = "";
        const natStr = r.nat ? t("history.nat_marker") : "";
        const n0 = teamNames[0], n1 = teamNames[1];
        text += t("history.round_header", {num: r.round_num, nat: natStr}) + "\n";
        text += t("history.trump_line", {suit: r.trump, player: r.declaring_player, teamname: teamNames[r.declaring_team]}) + "\n\n";

        if (r.roem_records && r.roem_records.length > 0) {
            text += t("history.roem_heading") + "\n";
            for (const entry of r.roem_records) {
                const desc = entry.items.map(([d, p]) => `${d} +${p}`).join(", ");
                text += t("history.roem_team", {teamname: teamNames[entry.team], detail: desc, total: entry.total}) + "\n";
            }
            text += t("history.roem_totals", {name0: n0, name1: n1, r0: r.roem_pts[0], r1: r.roem_pts[1]}) + "\n\n";
        } else {
            text += t("history.roem_none") + "\n\n";
        }

        text += t("history.tricks_heading") + "\n";
        for (let i = 0; i < r.tricks.length; i++) {
            const trick = r.tricks[i];
            const cards = trick.cards.map(([name, card]) => `${name}: ${card}`).join("  ");
            const bonus = i === 7 ? t("history.last_trick_bonus") : "";
            text += t("history.trick_line", {num: i + 1, cards}) + "\n";
            text += t("history.trick_winner", {name: trick.winner, pts: trick.pts, bonus}) + "\n";
        }
        text += "\n";

        text += t("history.scoring_heading") + "\n";
        text += t("history.trick_pts", {name0: n0, name1: n1, t0: r.trick_pts[0], t1: r.trick_pts[1]}) + "\n";
        if (r.roem_records && r.roem_records.length > 0) {
            text += t("history.roem_pts", {name0: n0, name1: n1, r0: r.roem_pts[0], r1: r.roem_pts[1]}) + "\n";
        }
        if (r.nat) {
            text += t("history.nat_line", {teamname: teamNames[r.declaring_team], othername: teamNames[1 - r.declaring_team]}) + "\n";
        }
        text += t("history.round_total", {name0: n0, name1: n1, t0: r.round_pts[0], t1: r.round_pts[1]}) + "\n";
        text += t("history.running", {name0: n0, name1: n1, s0: r.scores_after[0], s1: r.scores_after[1]}) + "\n";

        detailEl.textContent = text;
    }

    listEl.querySelectorAll(".history-list-item").forEach(el => {
        el.addEventListener("click", () => showRound(parseInt(el.dataset.idx)));
    });

    showRound(rounds.length - 1);
    document.getElementById("history-overlay").classList.add("active");
});

function openHistory() {
    socket.emit("get_history");
}

/* ─── Next round ──────────────────────────────────────────────────────── */

function nextRound() {
    document.getElementById("nextround-banner").classList.remove("active");
    socket.emit("next_round");
}

/* ─── New game ────────────────────────────────────────────────────────── */

function newGame() {
    clearTrickArea();
    clearDeclaringHighlight();
    for (const id of SEAT_CARD_IDS)
        document.getElementById(id).innerHTML = "";
    document.getElementById("trump-label").textContent = t("score.trump_default");
    const trumpInd = document.getElementById("trump-card-indicator");
    if (trumpInd) trumpInd.classList.remove("active");
    updateScores([0, 0]);
    updateRoundScores([0, 0], [0, 0]);
    socket.emit("new_game");
}

/* ─── Leave game ──────────────────────────────────────────────────────── */

function leaveGame() {
    document.getElementById("leave-overlay").classList.add("active");
}

function confirmLeave() {
    document.getElementById("leave-overlay").classList.remove("active");
    clearSession();
    socket.emit("leave_game");
}

socket.on("game_left", data => {
    // Return everyone to the lobby
    clearSession();
    document.getElementById("leave-game-btn").style.display = "none";
    document.getElementById("new-game-btn").style.display = "none";
    document.getElementById("nextround-banner").classList.remove("active");
    document.getElementById("gameover-overlay").classList.remove("active");
    document.getElementById("paused-overlay").classList.remove("active");
    document.getElementById("lobby-waiting").style.display = "none";
    document.getElementById("lobby-name-section").style.display = "block";
    document.getElementById("lobby-actions").style.display = "block";
    document.getElementById("lobby-overlay").classList.add("active");
    showLobbyError(t("msg.game_left", {name: data.name}));
    roomCode = null;
    isCreator = false;
});

/* ─── Chat ────────────────────────────────────────────────────────────── */

let chatOpen = false;
let chatUnread = 0;

function escapeHtml(s) {
    return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function toggleChat() {
    chatOpen ? closeChat() : openChat();
}

function openChat() {
    chatOpen = true;
    chatUnread = 0;
    document.getElementById("chat-panel").classList.add("open");
    document.getElementById("chat-badge").classList.remove("visible");
    document.getElementById("chat-input").focus();
    const msgs = document.getElementById("chat-messages");
    msgs.scrollTop = msgs.scrollHeight;
}

function closeChat() {
    chatOpen = false;
    document.getElementById("chat-panel").classList.remove("open");
}

function sendChat() {
    const input = document.getElementById("chat-input");
    const text = input.value.trim();
    if (!text) return;
    socket.emit("chat_message", {text});
    input.value = "";
    input.focus();
}

socket.on("chat_message", data => {
    const msgs = document.getElementById("chat-messages");
    const team = SEAT_TEAMS[data.seat];
    const div = document.createElement("div");
    div.className = `chat-msg team-${team}`;
    div.innerHTML = `<span class="chat-sender">${escapeHtml(data.name)}:</span>${escapeHtml(data.text)}`;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;

    if (!chatOpen) {
        chatUnread++;
        const badge = document.getElementById("chat-badge");
        badge.textContent = chatUnread > 99 ? "99+" : String(chatUnread);
        badge.classList.add("visible");
    }
});

/* ─── Initialization ──────────────────────────────────────────────────── */

// Apply saved language on page load
(function init() {
    setLang(getLang());
})();
