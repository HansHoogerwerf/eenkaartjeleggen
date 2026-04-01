/* ─── Lobby ───────────────────────────────────────────────────────────── */

function createRoom() {
    const name = document.getElementById("lobby-name").value.trim();
    if (!name) {
        showLobbyError(t("error.enter_name_join"), { sticky: true });
        document.getElementById("lobby-name").focus();
        return;
    }
    socket.emit("create_room", {name});
}

function peekRoom() {
    const name = document.getElementById("lobby-name").value.trim();
    if (!name) {
        showLobbyError(t("error.enter_name_join"), { sticky: true });
        document.getElementById("lobby-name").focus();
        return;
    }
    const code = cleanRoomCode(document.getElementById("join-code").value);
    document.getElementById("join-code").value = code;
    if (!code || code.length < 4) {
        showLobbyError(t("error.enter_code"));
        return;
    }
    socket.emit("peek_room", {code});
}

function joinRoom(seat) {
    const name = document.getElementById("lobby-name").value.trim();
    if (!name) {
        showLobbyError(t("error.enter_name_join"), { sticky: true });
        document.getElementById("lobby-name").focus();
        return;
    }
    const code = cleanRoomCode(document.getElementById("join-code").value);
    socket.emit("join_room", {code, name, seat});
}

function cleanRoomCode(value) {
    return (value || "").toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 4);
}

function bindLobbyInputUx() {
    const nameEl = document.getElementById("lobby-name");
    const codeEl = document.getElementById("join-code");

    if (nameEl) {
        nameEl.addEventListener("keydown", (event) => {
            if (event.key === "Enter") {
                const code = document.getElementById("join-code");
                if (code && code.value.trim()) peekRoom(); else createRoom();
            }
        });
        nameEl.addEventListener("input", () => {
            if (nameEl.value.trim()) clearLobbyError();
        });
    }

    if (codeEl) {
        const normalizeCode = () => {
            codeEl.value = cleanRoomCode(codeEl.value);
        };
        codeEl.addEventListener("input", normalizeCode);
        codeEl.addEventListener("paste", () => setTimeout(normalizeCode, 0));
        codeEl.addEventListener("keydown", (event) => {
            if (event.key === "Enter") peekRoom();
        });
    }
}

function setRoomUrl(code) {
    const url = new URL(window.location.href);
    url.searchParams.set("room", code);
    history.replaceState(null, "", url.toString());
}

function shareOrCopy() {
    const code = (document.getElementById("lobby-code").textContent || "").trim();
    if (!code) return;
    const url = new URL(window.location.href);
    url.searchParams.set("room", code);
    const shareUrl = url.toString();
    const btn = document.getElementById("lobby-share-btn");

    // On mobile (touch/coarse pointer), use the native share sheet
    const isMobile = window.matchMedia("(pointer: coarse)").matches;
    if (isMobile && navigator.share) {
        navigator.share({ title: document.title, url: shareUrl }).catch(() => {});
        return;
    }

    // On desktop, copy the URL and give feedback
    const onCopied = () => {
        if (!btn) return;
        const orig = t("lobby.share_btn");
        btn.textContent = t("lobby.link_copied");
        setTimeout(() => { btn.textContent = orig; }, 1300);
    };
    const fallback = () => {
        const input = document.createElement("input");
        input.value = shareUrl;
        document.body.appendChild(input);
        input.select();
        document.execCommand("copy");
        document.body.removeChild(input);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(shareUrl).then(onCopied).catch(() => { fallback(); onCopied(); });
        return;
    }
    fallback();
    onCopied();
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
let selectedAiStrength = "expert";
let selectedRulesVariant = "rotterdam";
let hasPendingRulesVariantSelection = false;

function selectMode(mode) {
    selectedMode = mode;
    document.querySelectorAll(".mode-btn").forEach(btn => {
        const isActive = btn.dataset.mode === mode;
        btn.setAttribute("aria-pressed", isActive ? "true" : "false");
        if (isActive) {
            btn.className = "btn-declare mode-btn active";
        } else {
            btn.className = "btn-pass mode-btn";
        }
    });
    document.getElementById("score-limit-input").style.display =
        mode === "score_limit" ? "block" : "none";
}

function selectAiStrength(level) {
    selectedAiStrength = level;
    document.querySelectorAll(".ai-btn").forEach(btn => {
        const isActive = btn.dataset.strength === level;
        btn.setAttribute("aria-pressed", isActive ? "true" : "false");
        if (isActive) {
            btn.className = "btn-declare ai-btn active";
        } else {
            btn.className = "btn-pass ai-btn";
        }
    });
}

function selectRulesVariant(variant, options = {}) {
    const { fromLobby = false, keepPending = false } = options;
    selectedRulesVariant = variant;

    if (isCreator) {
        if (fromLobby) {
            if (!keepPending) hasPendingRulesVariantSelection = false;
        } else {
            hasPendingRulesVariantSelection = true;
        }
    }

    document.querySelectorAll(".rules-btn").forEach(btn => {
        const isActive = btn.dataset.rules === variant;
        btn.setAttribute("aria-pressed", isActive ? "true" : "false");
        if (isActive) {
            btn.className = "btn-declare rules-btn active";
        } else {
            btn.className = "btn-pass rules-btn";
        }
    });
}

function startGame() {
    const data = { mode: selectedMode, ai_strength: selectedAiStrength, rules_variant: selectedRulesVariant };
    if (selectedMode === "score_limit") {
        const el = document.getElementById("mode-score-limit");
        data.score_limit = el ? (parseInt(el.value) || 500) : 500;
    }
    const t0 = document.getElementById("team0-name");
    const t1 = document.getElementById("team1-name");
    data.team_names = [
        t0 ? (t0.value.trim() || t0.placeholder || "Team 0") : "Team 0",
        t1 ? (t1.value.trim() || t1.placeholder || "Team 1") : "Team 1",
    ];
    socket.emit("start_game", data);
}

function showLobbyError(msg, { sticky = false } = {}) {
    const el = document.getElementById("lobby-error");
    el.textContent = msg;
    el.style.display = "block";
    if (!sticky) setTimeout(() => el.style.display = "none", 4000);
}

function clearLobbyError() {
    const el = document.getElementById("lobby-error");
    el.style.display = "none";
}

function showWaitingRoom(lobby) {
    document.getElementById("lobby-reconnecting").style.display = "none";
    document.getElementById("lobby-actions").style.display = "none";
    document.getElementById("lobby-name-section").style.display = "none";
    document.getElementById("lobby-seat-picker").style.display = "none";
    document.getElementById("lobby-waiting").style.display = "block";
    document.getElementById("lobby-code").textContent = lobby.code;
    setRoomUrl(lobby.code);
    updateLobbySeats(lobby);
}

function cancelLobby() {
    clearSession();
    history.replaceState(null, "", window.location.pathname);
    socket.emit("leave_room");
    document.getElementById("lobby-waiting").style.display = "none";
    document.getElementById("lobby-name-section").style.display = "block";
    document.getElementById("lobby-actions").style.display = "block";
}

function updateLobbySeats(lobby) {
    if (lobby && typeof lobby.ai_strength === "string") {
        if (["beginner", "advanced", "expert", "expert_v2", "expert_v3"].includes(lobby.ai_strength)) {
            selectedAiStrength = lobby.ai_strength;
        }
    }
    if (lobby && typeof lobby.rules_variant === "string") {
        if (["rotterdam", "amsterdam"].includes(lobby.rules_variant)) {
            if (!isCreator || !hasPendingRulesVariantSelection) {
                selectRulesVariant(lobby.rules_variant, { fromLobby: true });
            }
        }
    }

    const container = document.getElementById("lobby-seats");
    container.innerHTML = "";
    const teamPlayerNames = {0: [], 1: []};
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
        if (seat.is_human) teamPlayerNames[seat.team].push(seat.name);
    }
    const t0el = document.getElementById("team0-name");
    const t1el = document.getElementById("team1-name");
    if (t0el) t0el.placeholder = teamPlayerNames[0].join(" & ") || "Team 0";
    if (t1el) t1el.placeholder = teamPlayerNames[1].join(" & ") || "Team 1";

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

    const aiPicker = document.getElementById("lobby-ai-picker");
    const aiDisplay = document.getElementById("lobby-ai-display");
    if (aiPicker) {
        aiPicker.style.display = isCreator ? "block" : "none";
    }
    if (aiDisplay) {
        if (!isCreator) {
            aiDisplay.style.display = "block";
            document.getElementById("lobby-ai-text").textContent = t("ai." + selectedAiStrength);
        } else {
            aiDisplay.style.display = "none";
        }
    }

    const rulesPicker = document.getElementById("lobby-rules-picker");
    const rulesDisplay = document.getElementById("lobby-rules-display");
    if (rulesPicker) {
        rulesPicker.style.display = isCreator ? "block" : "none";
    }
    if (rulesDisplay) {
        if (!isCreator) {
            rulesDisplay.style.display = "block";
            document.getElementById("lobby-rules-text").textContent = t("rules." + selectedRulesVariant);
        } else {
            rulesDisplay.style.display = "none";
        }
    }

    if (isCreator) {
        selectAiStrength(selectedAiStrength);
        selectRulesVariant(selectedRulesVariant, { fromLobby: true, keepPending: true });
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
    isCreator = data.is_creator === true;
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

    // Show gameover New Game button only for creator; Leave/Close button for everyone
    const creatorDisplay = isCreator ? "inline-block" : "none";
    document.getElementById("gameover-newgame-btn").style.display = creatorDisplay;
    document.getElementById("gameover-close-btn").style.display = "inline-block";
    document.getElementById("gameover-banner-newgame-btn").style.display = creatorDisplay;
    document.getElementById("leave-game-btn").style.display = "inline-block";
});

/* ─── Game socket events ──────────────────────────────────────────────── */

socket.on("deal_done", data => {
    document.getElementById("nextround-banner").classList.remove("active");
    dismissGameover();
    clearBidBadges();
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

    const indicator = document.getElementById("trump-card-indicator");
    const card = document.getElementById("trump-card");
    if (indicator && card) {
        card.className = isRed(data.suit) ? "card red" : "card";
        card.innerHTML = `<span>${data.suit}</span><span>${data.suit}</span>`;
        indicator.classList.add("active");
    }
});

socket.on("bid", data => {
    showBidBadge(data.player_idx, data.trump !== null);
});

socket.on("trump_set", data => {

    // Bidding is complete — remove all pass/declare badges
    clearBidBadges();

    // Highlight declaring player's seat label
    clearDeclaringHighlight();
    if (data.declaring_player_idx !== undefined) {
        showDeclaringHighlight(data.declaring_player_idx, data.trump);
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
    // Start slide animation toward the winner during the backend pause
    if (data.winner_idx !== undefined)
        clearTrickArea(data.winner_idx);
});

socket.on("trick_cleared", data => {
    // Instant cleanup — the slide animation already ran on trick_won
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
    const winner = teamNames[data.winner];
    document.getElementById("gameover-winner").textContent =
        t("modal.game_over_winner", {winner});
    document.getElementById("gameover-scores").textContent =
        t("modal.game_over_scores", {team0: teamNames[0], s0: s[0], team1: teamNames[1], s1: s[1]});
    document.getElementById("gameover-banner-msg").textContent =
        t("modal.game_over_winner", {winner});
    document.getElementById("gameover-overlay").classList.add("active");
});

function dismissGameover() {
    document.getElementById("gameover-overlay").classList.remove("active");
    document.getElementById("gameover-banner").classList.remove("active");
}

function showGameoverBanner() {
    document.getElementById("gameover-banner").classList.add("active");
}

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

    msgDiv.innerHTML = "";
    const mainMsg = document.createElement("div");
    mainMsg.textContent = data.forced ? t("bid.forced") : t("bid.optional");
    msgDiv.appendChild(mainMsg);
    if (data.leader_name) {
        const leaderMsg = document.createElement("div");
        leaderMsg.className = "bid-leader-note";
        leaderMsg.textContent = t("bid.leader", {name: data.leader_name});
        msgDiv.appendChild(leaderMsg);
    }

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

socket.on("request_forced_suit", () => {
    const overlay = document.getElementById("bid-overlay");
    const suitDiv = document.getElementById("bid-suit");
    const msgDiv = document.getElementById("bid-msg");
    const btnsDiv = document.getElementById("bid-buttons");

    suitDiv.className = "bid-suit";
    suitDiv.textContent = "";

    msgDiv.textContent = t("bid.pick_suit");

    btnsDiv.innerHTML = "";
    for (const suit of ["♣", "♦", "♥", "♠"]) {
        const btn = document.createElement("button");
        btn.className = `btn-suit-pick ${isRed(suit) ? "red" : "black"}`;
        btn.textContent = `${suit} ${tSuit(suit)}`;
        btn.onclick = () => {
            socket.emit("forced_suit_response", {suit});
            overlay.classList.remove("active");
        };
        btnsDiv.appendChild(btn);
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
    const creatorDisplay = isCreator ? "inline-block" : "none";
    document.getElementById("gameover-newgame-btn").style.display = creatorDisplay;
    document.getElementById("gameover-close-btn").style.display = "inline-block";
    document.getElementById("gameover-banner-newgame-btn").style.display = creatorDisplay;
    updateSeatLabels();
    updateScores(data.scores);
    updateRoundScores(data.cur_tricks, data.cur_roem);

    if (data.hand) renderMyHand(data.hand, []);
    if (data.card_counts) renderOtherCards(data.card_counts);

    // Restore trump indicator
    if (data.trump) {

        clearDeclaringHighlight();
        if (data.declaring_player_idx !== null && data.declaring_player_idx !== undefined) {
            showDeclaringHighlight(data.declaring_player_idx, data.trump);
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
        listEl.innerHTML = `<p class="history-empty">${t("history.empty")}</p>`;
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

socket.on("game_aborted", data => {
    // Game was interrupted (e.g. player disconnect timeout) — return to lobby
    document.getElementById("leave-game-btn").style.display = "none";
    document.getElementById("nextround-banner").classList.remove("active");
    document.getElementById("gameover-banner").classList.remove("active");
    document.getElementById("paused-overlay").classList.remove("active");
    document.getElementById("lobby-name-section").style.display = "none";
    document.getElementById("lobby-actions").style.display = "none";
    document.getElementById("lobby-waiting").style.display = "block";
    document.getElementById("lobby-overlay").classList.add("active");
    const msg = data.key ? t(data.key) : "The game was aborted.";
    showLobbyError(msg);
});

socket.on("game_left", data => {
    // Return everyone to the lobby
    clearSession();
    history.replaceState(null, "", window.location.pathname);
    document.getElementById("leave-game-btn").style.display = "none";
    document.getElementById("nextround-banner").classList.remove("active");
    dismissGameover();
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
    const panel = document.getElementById("chat-panel");
    const badge = document.getElementById("chat-badge");
    const toggleBtn = document.getElementById("chat-toggle-btn");
    panel.classList.add("open");
    panel.setAttribute("aria-hidden", "false");
    if (toggleBtn) toggleBtn.setAttribute("aria-expanded", "true");
    badge.classList.remove("visible");
    badge.textContent = "";
    badge.setAttribute("aria-hidden", "true");
    document.getElementById("chat-input").focus();
    const msgs = document.getElementById("chat-messages");
    msgs.scrollTop = msgs.scrollHeight;
}

function closeChat() {
    chatOpen = false;
    const panel = document.getElementById("chat-panel");
    const toggleBtn = document.getElementById("chat-toggle-btn");
    panel.classList.remove("open");
    panel.setAttribute("aria-hidden", "true");
    if (toggleBtn) toggleBtn.setAttribute("aria-expanded", "false");
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
        badge.setAttribute("aria-hidden", "false");
    }
});

/* ─── Modal keyboard accessibility ────────────────────────────────────── */

const MODAL_OVERLAY_SELECTOR = ".modal-overlay";
const FOCUSABLE_SELECTOR = [
    "button:not([disabled])",
    "[href]",
    "input:not([disabled]):not([type='hidden'])",
    "select:not([disabled])",
    "textarea:not([disabled])",
    "[tabindex]:not([tabindex='-1'])",
].join(", ");
const modalFocusRestore = new WeakMap();
let modalObserver = null;

function isVisibleForFocus(el) {
    return !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
}

function getActiveModalOverlays() {
    return Array.from(document.querySelectorAll(`${MODAL_OVERLAY_SELECTOR}.active`))
        .filter(isVisibleForFocus);
}

function getTopActiveModalOverlay() {
    const modals = getActiveModalOverlays();
    return modals.length > 0 ? modals[modals.length - 1] : null;
}

function getFocusableWithin(container) {
    return Array.from(container.querySelectorAll(FOCUSABLE_SELECTOR))
        .filter(el => isVisibleForFocus(el) && el.getAttribute("aria-hidden") !== "true");
}

function focusFirstInModal(modal) {
    const focusable = getFocusableWithin(modal);
    if (focusable.length > 0) {
        focusable[0].focus();
        return;
    }

    const fallback = modal.querySelector(".modal") || modal;
    if (!fallback.hasAttribute("tabindex")) fallback.setAttribute("tabindex", "-1");
    fallback.focus();
}

function closeTopModalWithEscape(modal) {
    const closeBtn = Array.from(modal.querySelectorAll("[data-modal-close]"))
        .find(btn => !btn.disabled && isVisibleForFocus(btn));
    if (!closeBtn) return false;
    closeBtn.click();
    return true;
}

function handleModalOpened(modal) {
    if (!modalFocusRestore.has(modal)) {
        const active = document.activeElement;
        if (active instanceof HTMLElement && !modal.contains(active)) {
            modalFocusRestore.set(modal, active);
        }
    }
    setTimeout(() => {
        const topModal = getTopActiveModalOverlay();
        const active = document.activeElement;
        if (topModal === modal && !(active instanceof HTMLElement && modal.contains(active))) {
            focusFirstInModal(modal);
        }
    }, 0);
}

function handleModalClosed(modal) {
    const restoreEl = modalFocusRestore.get(modal);
    modalFocusRestore.delete(modal);
    if (restoreEl instanceof HTMLElement && document.contains(restoreEl)) {
        restoreEl.focus();
    }
}

function hadClass(oldValue, className) {
    return (oldValue || "").split(/\s+/).includes(className);
}

function handleModalKeydown(event) {
    const modal = getTopActiveModalOverlay();
    if (!modal) return;

    if (event.key === "Escape") {
        if (closeTopModalWithEscape(modal)) {
            event.preventDefault();
            event.stopPropagation();
        }
        return;
    }

    if (event.key !== "Tab") return;
    const focusable = getFocusableWithin(modal);
    if (focusable.length === 0) {
        event.preventDefault();
        focusFirstInModal(modal);
        return;
    }

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;

    if (!(active instanceof HTMLElement) || !modal.contains(active)) {
        event.preventDefault();
        first.focus();
        return;
    }
    if (event.shiftKey && active === first) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
    }
}

function initModalKeyboardAccessibility() {
    modalObserver = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            if (mutation.type !== "attributes" || mutation.attributeName !== "class") continue;
            if (!(mutation.target instanceof HTMLElement)) continue;
            const modal = mutation.target;
            const isActive = modal.classList.contains("active");
            const wasActive = hadClass(mutation.oldValue, "active");
            if (isActive && !wasActive) handleModalOpened(modal);
            if (!isActive && wasActive) handleModalClosed(modal);
        }
    });

    document.querySelectorAll(MODAL_OVERLAY_SELECTOR).forEach((modal) => {
        modalObserver.observe(modal, {
            attributes: true,
            attributeFilter: ["class"],
            attributeOldValue: true,
        });
    });

    document.addEventListener("keydown", handleModalKeydown);
}

/* ─── Initialization ──────────────────────────────────────────────────── */

// Apply saved language on page load
(function init() {
    setLang(getLang());
    initModalKeyboardAccessibility();
    bindLobbyInputUx();
    const urlCode = new URLSearchParams(window.location.search).get("room");
    if (urlCode) {
        const codeEl = document.getElementById("join-code");
        if (codeEl) codeEl.value = cleanRoomCode(urlCode);
    }
    const nameEl = document.getElementById("lobby-name");
    if (nameEl) {
        const savedName = loadPlayerName();
        if (savedName) nameEl.value = savedName;
        nameEl.focus();
    }
})();
