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
