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

function renderMyHand(cards, legal) {
    const el = document.getElementById(cardContainerId(mySeat));
    const legalSet = new Set(legal || []);
    el.innerHTML = cards.map(c => {
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
    slot.className = "trick-slot filled";
    slot.style.color = isRed(card.suit) ? "var(--red-suit)" : "var(--black-suit)";
    slot.innerHTML = `<span>${card.rank}</span><span>${card.suit}</span>`;
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
    showWaitingRoom(data.lobby);
});

socket.on("room_joined", data => {
    roomCode = data.code;
    mySeat = data.seat;
    isCreator = false;
    showWaitingRoom(data.lobby);
});

socket.on("room_peeked", data => {
    showSeatPicker(data.lobby);
});

socket.on("join_error", data => {
    // Server sends translation key
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

    // Show New Game button only for creator
    document.getElementById("new-game-btn").style.display = isCreator ? "inline-block" : "none";
    document.getElementById("gameover-newgame-btn").style.display = isCreator ? "inline-block" : "none";
});

/* ─── Game socket events ──────────────────────────────────────────────── */

socket.on("deal_done", data => {
    document.getElementById("nextround-banner").classList.remove("active");
    clearDeclaringHighlight();
    const trumpInd = document.getElementById("trump-card-indicator");
    if (trumpInd) trumpInd.classList.remove("active");
    clearTrickArea();
    if (data.my_seat !== undefined) mySeat = data.my_seat;
    renderMyHand(data.hand, []);
    if (data.card_counts) renderOtherCards(data.card_counts);
    updateRoundScores([0, 0], [0, 0]);
});

socket.on("trump_offered", data => {
    const color = isRed(data.suit) ? "red" : "var(--accent)";
    document.getElementById("trump-label").innerHTML =
        `<span style="color:${color}">${t("score.offered", {suit: data.suit, name: tSuit(data.suit), round: data.round_num})}</span>`;
});

socket.on("bid", () => {});

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

    if (data.hand) renderMyHand(data.hand, []);
});

socket.on("trick_played", data => {
    showCardInTrick(data.player_idx, data.card);
    if (data.hand) renderMyHand(data.hand, currentLegal);
    if (data.card_counts) renderOtherCards(data.card_counts);
});

socket.on("trick_won", data => {
    if (data.cur_tricks && data.cur_roem)
        updateRoundScores(data.cur_tricks, data.cur_roem);
});

socket.on("trick_cleared", () => {
    clearTrickArea();
    currentLegal = [];
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
    appendLog(t("log.disconnected", {name: data.name}), "nat");
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
    updateSeatLabels();
    updateScores(data.scores);
    updateRoundScores(data.cur_tricks, data.cur_roem);

    if (data.hand) renderMyHand(data.hand, []);
    if (data.card_counts) renderOtherCards(data.card_counts);
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

/* ─── Initialization ──────────────────────────────────────────────────── */

// Apply saved language on page load
(function init() {
    setLang(getLang());
})();
