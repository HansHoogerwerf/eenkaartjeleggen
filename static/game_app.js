/* ─── Lobby ───────────────────────────────────────────────────────────── */

function createRoom() {
    const name = document.getElementById("lobby-name").value.trim() || t("lobby.name_placeholder");
    socket.emit("create_room", {name});
}

function peekRoom() {
    const code = cleanRoomCode(document.getElementById("join-code").value);
    document.getElementById("join-code").value = code;
    if (!code || code.length < 4) {
        showLobbyError(t("error.enter_code"));
        return;
    }
    socket.emit("peek_room", {code});
}

function joinRoom(seat) {
    const name = document.getElementById("lobby-name").value.trim() || t("lobby.name_placeholder");
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
            if (event.key === "Enter") createRoom();
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

function copyRoomCode() {
    const code = (document.getElementById("lobby-code").textContent || "").trim();
    if (!code) return;
    const btn = document.getElementById("lobby-copy-code");
    const copiedTxt = t("lobby.copied");
    const fallbackCopy = () => {
        const input = document.createElement("input");
        input.value = code;
        document.body.appendChild(input);
        input.select();
        document.execCommand("copy");
        document.body.removeChild(input);
    };

    const onCopied = () => {
        if (!btn) return;
        btn.textContent = copiedTxt;
        setTimeout(() => {
            btn.textContent = t("lobby.copy_code");
        }, 1300);
    };

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(code).then(onCopied).catch(() => {
            fallbackCopy();
            onCopied();
        });
        return;
    }

    fallbackCopy();
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

function selectAiStrength(level) {
    selectedAiStrength = level;
    document.querySelectorAll(".ai-btn").forEach(btn => {
        if (btn.dataset.strength === level) {
            btn.className = "btn-declare ai-btn active";
        } else {
            btn.className = "btn-pass ai-btn";
        }
    });
}

function startGame() {
    const data = { mode: selectedMode, ai_strength: selectedAiStrength };
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
    if (lobby && typeof lobby.ai_strength === "string") {
        if (["beginner", "advanced", "expert"].includes(lobby.ai_strength)) {
            selectedAiStrength = lobby.ai_strength;
        }
    }

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
    if (isCreator) {
        selectAiStrength(selectedAiStrength);
    }
}

function applyCreatorState() {
    const lobbyStart = document.getElementById("lobby-start-btn");
    const lobbyWait = document.getElementById("lobby-wait-msg");
    const newGameBtn = document.getElementById("new-game-btn");
    const gameOverNewGameBtn = document.getElementById("gameover-newgame-btn");
    const nextRoundBtn = document.getElementById("nextround-btn");
    const nextRoundWait = document.getElementById("nextround-wait");

    if (lobbyStart) lobbyStart.style.display = isCreator ? "inline-block" : "none";
    if (lobbyWait) lobbyWait.style.display = isCreator ? "none" : "block";
    if (newGameBtn) newGameBtn.style.display = isCreator ? "inline-block" : "none";
    if (gameOverNewGameBtn) gameOverNewGameBtn.style.display = isCreator ? "inline-block" : "none";
    if (nextRoundBtn) nextRoundBtn.style.display = isCreator ? "inline-block" : "none";
    if (nextRoundWait) nextRoundWait.style.display = isCreator ? "none" : "inline";
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

socket.on("host_migrated", data => {
    if (data && data.seat !== undefined) {
        isCreator = Number(data.seat) === Number(mySeat);
    }
    applyCreatorState();
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
    applyCreatorState();

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
    bindLobbyInputUx();
    const nameEl = document.getElementById("lobby-name");
    if (nameEl) nameEl.focus();
})();
