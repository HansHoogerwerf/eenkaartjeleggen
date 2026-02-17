/* ─── Klaverjassen i18n — English + Dutch ─────────────────────────────── */

const TRANSLATIONS = {
    en: {
        // ── Page title ──
        "page.title": "Klaverjassen — Rotterdam rules",

        // ── Lobby ──
        "lobby.heading": "Klaverjassen",
        "lobby.subtitle": "Rotterdam rules — 1 to 4 players",
        "lobby.your_name": "Your name:",
        "lobby.name_placeholder": "Player",
        "lobby.create_game": "Create Game",
        "lobby.or": "— or —",
        "lobby.code_placeholder": "CODE",
        "lobby.join": "Join",
        "lobby.choose_seat": "Choose your seat:",
        "lobby.back": "Back",
        "lobby.room_code": "Room code:",
        "lobby.start_game": "Start Game",
        "lobby.team_names": "Team names:",
        "lobby.waiting_host": "Waiting for the host to start...",

        // ── Seat names ──
        "seat.south": "South",
        "seat.west": "West",
        "seat.north": "North",
        "seat.east": "East",
        "seat.you_suffix": " (you)",
        "seat.open": "Open",
        "seat.ai": "AI",

        // ── Suit names ──
        "suit.♣": "Clubs",
        "suit.♦": "Diamonds",
        "suit.♥": "Hearts",
        "suit.♠": "Spades",

        // ── Teams ──
        "team.0": "Team 0",
        "team.1": "Team 1",

        // ── Score panel ──
        "score.team0": "{name}: {score}",
        "score.team1": "{name}: {score}",
        "score.trump_default": "Trump: —",
        "score.trump": "Trump: {suit}  ({player} declares)",
        "score.offered": "Offered: {suit} ({name})  round {round}",
        "score.round_team0": "Round  {name}: {roem} roem + {tricks} tricks = {total}",
        "score.round_team1": "{name}: {roem} roem + {tricks} tricks = {total}",
        "score.round_default0": "Round  —",
        "score.round_default1": "—",

        // ── Buttons ──
        "btn.hands": "Hands",
        "btn.history": "History",
        "btn.new_game": "New Game",
        "btn.next_round": "Next Round",
        "btn.close": "Close",

        // ── Game mode ──
        "mode.label": "Game mode:",
        "mode.score_limit": "Score Limit",
        "mode.boom": "Boom (16 rounds)",
        "mode.free_play": "Free Play",
        "mode.target_score": "Target score:",

        // ── Next Round modal ──
        "modal.round_finished": "Round Finished",
        "msg.waiting_host_round": "Waiting for host to start next round...",

        // ── Bid modal ──
        "bid.heading": "Trump Declaration",
        "bid.forced": "You are forced to declare this suit!",
        "bid.optional": "Declare this suit as trump?",
        "bid.declare": "Declare",
        "bid.pass": "Pass",

        // ── Game Over modal ──
        "modal.game_over": "Game Over",
        "modal.game_over_msg": "{winner} wins!\n\nFinal scores:\n  {team0}: {s0}    {team1}: {s1}",

        // ── Paused modal ──
        "modal.paused": "Game Paused",

        // ── Hands modal ──
        "modal.hands": "Hands — Current Round",
        "hands.empty": "No tricks played yet this round.",
        "hands.trick_header": "Trick {num}  —  {winner} wins  (+{pts} pts)",

        // ── History modal ──
        "modal.history_rounds": "Rounds",
        "modal.history_details": "Details",
        "history.empty": "No completed rounds yet.",
        "history.select_round": "Select a round...",
        "history.nat_marker": "  *** NAT ***",
        "history.round_header": "Round {num}{nat}",
        "history.trump_line": "Trump: {suit}  –  declared by {player} ({teamname})",
        "history.roem_heading": "Roem:",
        "history.roem_none": "Roem: none",
        "history.roem_team": "  {teamname}: {detail}  [+{total}]",
        "history.roem_totals": "  {name0} total: {r0}   {name1} total: {r1}",
        "history.tricks_heading": "Tricks:",
        "history.trick_line": "  Trick {num}: {cards}",
        "history.trick_winner": "    Winner: {name}  (+{pts} pts){bonus}",
        "history.last_trick_bonus": "  (+10 last trick bonus)",
        "history.scoring_heading": "Round scoring:",
        "history.trick_pts": "  Trick pts  — {name0}: {t0}   {name1}: {t1}",
        "history.roem_pts": "  Roem pts   — {name0}: {r0}   {name1}: {r1}",
        "history.nat_line": "  NAT! {teamname} goes nat — {othername} takes everything",
        "history.round_total": "  Round total— {name0}: {t0}   {name1}: {t1}",
        "history.running": "  Running    — {name0}: {s0}   {name1}: {s1}",
        "history.round_summary": "Rd {num}  +{t0}/{t1}{nat}",
        "history.nat_short": " NAT",

        // ── Disconnect / Reconnect ──
        "log.disconnected": "{name} disconnected.",
        "log.reconnected": "{name} reconnected.",

        // ── Error messages (from server keys) ──
        "error.room_not_found": "Room not found.",
        "error.game_in_progress": "Game already in progress.",
        "error.invalid_seat": "Invalid seat.",
        "error.seat_taken": "That seat is already taken.",
        "error.room_full": "Room is full.",
        "error.only_creator": "Only the room creator can start the game.",
        "error.enter_code": "Enter a 4-character room code.",
        "error.waiting_reconnect": "Waiting for {name} to reconnect...",
    },

    nl: {
        // ── Paginatitel ──
        "page.title": "Klaverjassen — Rotterdamse regels",

        // ── Lobby ──
        "lobby.heading": "Klaverjassen",
        "lobby.subtitle": "Rotterdamse regels — 1 tot 4 spelers",
        "lobby.your_name": "Je naam:",
        "lobby.name_placeholder": "Speler",
        "lobby.create_game": "Spel Aanmaken",
        "lobby.or": "— of —",
        "lobby.code_placeholder": "CODE",
        "lobby.join": "Deelnemen",
        "lobby.choose_seat": "Kies je stoel:",
        "lobby.back": "Terug",
        "lobby.room_code": "Kamercode:",
        "lobby.start_game": "Spel Starten",
        "lobby.team_names": "Teamnamen:",
        "lobby.waiting_host": "Wachten tot de host het spel start...",

        // ── Stoelnamen ──
        "seat.south": "Zuid",
        "seat.west": "West",
        "seat.north": "Noord",
        "seat.east": "Oost",
        "seat.you_suffix": " (jij)",
        "seat.open": "Vrij",
        "seat.ai": "AI",

        // ── Kleurnamen ──
        "suit.♣": "Klaveren",
        "suit.♦": "Ruiten",
        "suit.♥": "Harten",
        "suit.♠": "Schoppen",

        // ── Teams ──
        "team.0": "Team 0",
        "team.1": "Team 1",

        // ── Scorepaneel ──
        "score.team0": "{name}: {score}",
        "score.team1": "{name}: {score}",
        "score.trump_default": "Troefkleur: —",
        "score.trump": "Troefkleur: {suit}  ({player} gaat)",
        "score.offered": "Aangeboden: {suit} ({name})  ronde {round}",
        "score.round_team0": "Ronde  {name}: {roem} roem + {tricks} slagen = {total}",
        "score.round_team1": "{name}: {roem} roem + {tricks} slagen = {total}",
        "score.round_default0": "Ronde  —",
        "score.round_default1": "—",

        // ── Knoppen ──
        "btn.hands": "Handen",
        "btn.history": "Geschiedenis",
        "btn.new_game": "Nieuw Spel",
        "btn.next_round": "Volgende Ronde",
        "btn.close": "Sluiten",

        // ── Spelmodus ──
        "mode.label": "Spelmodus:",
        "mode.score_limit": "Puntenlimiet",
        "mode.boom": "Boom (16 rondes)",
        "mode.free_play": "Vrij Spelen",
        "mode.target_score": "Doelscore:",

        // ── Volgende Ronde modal ──
        "modal.round_finished": "Ronde Afgelopen",
        "msg.waiting_host_round": "Wachten tot de host de volgende ronde start...",

        // ── Troefmodal ──
        "bid.heading": "Troef Verklaring",
        "bid.forced": "Je bent gedwongen om op deze kleur te gaan!",
        "bid.optional": "Op deze kleur als troef gaan?",
        "bid.declare": "Gaan",
        "bid.pass": "Passen",

        // ── Spel Afgelopen modal ──
        "modal.game_over": "Spel Afgelopen",
        "modal.game_over_msg": "{winner} wint!\n\nEindstand:\n  {team0}: {s0}    {team1}: {s1}",

        // ── Gepauzeerd modal ──
        "modal.paused": "Spel Gepauzeerd",

        // ── Handen modal ──
        "modal.hands": "Handen — Huidige Ronde",
        "hands.empty": "Nog geen slagen gespeeld deze ronde.",
        "hands.trick_header": "Slag {num}  —  {winner} wint  (+{pts} ptn)",

        // ── Geschiedenis modal ──
        "modal.history_rounds": "Rondes",
        "modal.history_details": "Details",
        "history.empty": "Nog geen afgeronde rondes.",
        "history.select_round": "Selecteer een ronde...",
        "history.nat_marker": "  *** NAT ***",
        "history.round_header": "Ronde {num}{nat}",
        "history.trump_line": "Troef: {suit}  –  {player} ({teamname}) gaat",
        "history.roem_heading": "Roem:",
        "history.roem_none": "Roem: geen",
        "history.roem_team": "  {teamname}: {detail}  [+{total}]",
        "history.roem_totals": "  {name0} totaal: {r0}   {name1} totaal: {r1}",
        "history.tricks_heading": "Slagen:",
        "history.trick_line": "  Slag {num}: {cards}",
        "history.trick_winner": "    Winnaar: {name}  (+{pts} ptn){bonus}",
        "history.last_trick_bonus": "  (+10 laatste slag bonus)",
        "history.scoring_heading": "Rondescore:",
        "history.trick_pts": "  Slagpunten — {name0}: {t0}   {name1}: {t1}",
        "history.roem_pts": "  Roempunten — {name0}: {r0}   {name1}: {r1}",
        "history.nat_line": "  NAT! {teamname} gaat nat — {othername} krijgt alles",
        "history.round_total": "  Rondetotaal— {name0}: {t0}   {name1}: {t1}",
        "history.running": "  Lopend     — {name0}: {s0}   {name1}: {s1}",
        "history.round_summary": "Rd {num}  +{t0}/{t1}{nat}",
        "history.nat_short": " NAT",

        // ── Verbinding verbroken / hersteld ──
        "log.disconnected": "{name} heeft de verbinding verbroken.",
        "log.reconnected": "{name} is opnieuw verbonden.",

        // ── Foutmeldingen (van serversleutels) ──
        "error.room_not_found": "Kamer niet gevonden.",
        "error.game_in_progress": "Spel is al bezig.",
        "error.invalid_seat": "Ongeldige stoel.",
        "error.seat_taken": "Die stoel is al bezet.",
        "error.room_full": "Kamer is vol.",
        "error.only_creator": "Alleen de maker van de kamer kan het spel starten.",
        "error.enter_code": "Voer een 4-karakter kamercode in.",
        "error.waiting_reconnect": "Wachten tot {name} opnieuw verbindt...",
    },
};

let currentLang = localStorage.getItem("klaverjas_lang") || "nl";

/**
 * Translate a key, optionally interpolating {placeholder} values.
 * Falls back to English, then to the raw key.
 */
function t(key, params) {
    let str = (TRANSLATIONS[currentLang] && TRANSLATIONS[currentLang][key])
           || TRANSLATIONS.en[key]
           || key;
    if (params) {
        for (const [k, v] of Object.entries(params)) {
            str = str.replace(new RegExp(`\\{${k}\\}`, "g"), v);
        }
    }
    return str;
}

/** Get translated seat name by index (0=South, 1=West, 2=North, 3=East). */
function tSeat(idx) {
    const keys = ["seat.south", "seat.west", "seat.north", "seat.east"];
    return t(keys[idx]);
}

/** Get the 4 seat names as an array in the current language. */
function tSeatPositions() {
    return [t("seat.south"), t("seat.west"), t("seat.north"), t("seat.east")];
}

/** Get translated suit name by symbol (♣, ♦, ♥, ♠). */
function tSuit(symbol) { return t("suit." + symbol); }

function getLang() { return currentLang; }

function setLang(lang) {
    currentLang = lang;
    localStorage.setItem("klaverjas_lang", lang);
    document.title = t("page.title");
    applyStaticTranslations();
    // Update the flag button active states
    document.querySelectorAll(".lang-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.lang === lang);
    });
}

/**
 * Apply translations to all elements with a data-i18n attribute.
 * Also updates placeholder attributes via data-i18n-placeholder.
 */
function applyStaticTranslations() {
    document.querySelectorAll("[data-i18n]").forEach(el => {
        el.textContent = t(el.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
        el.placeholder = t(el.dataset.i18nPlaceholder);
    });
}
