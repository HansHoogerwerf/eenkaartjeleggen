# Fix: Missing UI Updates After Short Disconnections

## Problem Statement

During short disconnections, the UI can miss updates for cards played by the round starter (and potentially other players), causing confusion. The player sees an incomplete trick state after reconnecting.

## Root Cause Analysis

### Current Reconnection Flow

1. **Server side** ([`server/game_flow.py:81-119`](../server/game_flow.py:81-119)):
   - `apply_reconnect()` builds a snapshot via `build_game_state_snapshot()`
   - Snapshot includes `trick_cards` dict with currently played cards
   - Emits `game_state_snapshot` to reconnecting client

2. **Snapshot building** ([`server/game_flow.py:20-75`](../server/game_flow.py:20-75)):
   - Line 42: `"trick_cards": {str(k): v for k, v in room.cur_trick_cards.items()}`
   - `room.cur_trick_cards` is populated when `trick_played` events occur (line 361)
   - **CRITICAL**: `room.cur_trick_cards` is cleared when `trick_won` fires (line 389)

3. **Client side** ([`static/game_app.js:823-923`](../static/game_app.js:823-923)):
   - Line 848: `clearTrickArea()` — wipes all trick slots before restoring
   - Lines 901-907: Restores `trick_cards` from snapshot
   - Line 905: Calls `showCardInTrick()` for each card in the snapshot

### The Race Condition

**Scenario causing missing cards:**

```
Time  Event                           Server State                Client State
----  ------------------------------  --------------------------  ---------------------------
T0    Player 0 plays card             cur_trick_cards[0] = card   Shows card in trick slot 0
T1    Player 1 plays card             cur_trick_cards[1] = card   Shows card in trick slot 1
T2    Client disconnects              cur_trick_cards = {0, 1}    [disconnected]
T3    Player 2 plays card             cur_trick_cards[2] = card   [disconnected]
T4    Player 3 plays card             cur_trick_cards[3] = card   [disconnected]
T5    trick_won fires                 cur_trick_cards.clear()     [disconnected]
      (winner determined)             cur_trick_cards = {}        [disconnected]
T6    Client reconnects               Snapshot sent:              Receives snapshot:
                                      trick_cards = {}            trick_cards = {}
                                                                  → Shows EMPTY trick area
T7    trick_cleared fires             (no state change)           clearTrickArea() called
                                                                  → Still empty
```

**The problem:** Between when the trick completes (all 4 cards played) and when `trick_cleared` fires (after `TRICK_CLEAR_DELAY` = 1.4s), there's a window where:
- The server has already cleared `cur_trick_cards` (on `trick_won`)
- The client hasn't seen the `trick_won` event (was disconnected)
- The reconnect snapshot contains an empty `trick_cards` dict
- The client renders an empty trick area, missing all 4 cards

### Why This Affects the Round Starter Most

The round starter's card is the **first** card played in each trick. If a disconnection happens mid-trick and the trick completes while disconnected:
1. All 4 cards (including the starter's) are cleared from `cur_trick_cards`
2. The reconnecting client sees none of them
3. The starter's card is most noticeable because it's the "anchor" card that defines the trick's lead suit

## Solution Design

### Option 1: Preserve Trick Cards Until Cleared (Recommended)

**Change:** Don't clear `cur_trick_cards` on `trick_won` — only clear it on `trick_cleared`.

**Rationale:**
- The trick cards remain visible on-screen until `trick_cleared` fires anyway
- Reconnecting clients need to see the completed trick during that 1.4s window
- Aligns server state lifetime with client UI lifetime

**Implementation:**

1. **Server change** ([`server/game_flow.py:380-391`](../server/game_flow.py:380-391)):
   ```python
   if event == "trick_won":
       winner_idx = data["winner_idx"]
       pts = data["pts"]
       room.cur_round_tricks.append({
           "cards": dict(room.cur_trick_cards),  # Snapshot for history
           "winner_idx": winner_idx,
           "winner_name": room.player_names().get(winner_idx, "?"),
           "pts": pts,
       })
       # DON'T clear cur_trick_cards here — wait for trick_cleared
       room.cur_tricks[:] = list(data["trick_pts"])
       room.cur_roem[:] = list(data["roem_pts"])
       send_data["cur_tricks"] = list(room.cur_tricks)
       send_data["cur_roem"] = list(room.cur_roem)
   ```

2. **Add `trick_cleared` handler** (new, after line 407):
   ```python
   if event == "trick_cleared":
       room.cur_trick_cards.clear()
       # No need to emit — clients already handle this event from main.py
       return
   ```

3. **Ensure `trick_cleared` is routed** ([`server/game_flow.py:283`](../server/game_flow.py:283)):
   - Currently `room_state()` has a catch-all `socketio.emit(event, send_data, room=room.code)` at line 407
   - `trick_cleared` will fall through to this, which is correct
   - Just need to add the clear logic before the emit

**Benefits:**
- Minimal code change (2 lines modified, ~5 lines added)
- Fixes the race condition completely
- No client changes needed
- Preserves existing event flow

**Risks:**
- `cur_trick_cards` now persists 1.4s longer per trick
- Memory impact negligible (4 card dicts × 8 tricks × N rooms)

### Option 2: Include Last Completed Trick in Snapshot

**Change:** Add `last_trick_cards` to snapshot, separate from `cur_trick_cards`.

**Rationale:**
- Keeps `cur_trick_cards` semantics unchanged (current in-progress trick only)
- Explicitly tracks the "just completed but not yet cleared" trick

**Implementation:**

1. **Add room state** ([`server/room_state.py:54`](../server/room_state.py:54)):
   ```python
   self.last_trick_cards: dict[int, dict] = {}  # Cards from most recent completed trick
   ```

2. **Update on `trick_won`** ([`server/game_flow.py:389`](../server/game_flow.py:389)):
   ```python
   if event == "trick_won":
       # ... existing code ...
       room.last_trick_cards = dict(room.cur_trick_cards)  # Preserve for reconnect
       room.cur_trick_cards.clear()
   ```

3. **Clear on `trick_cleared`**:
   ```python
   if event == "trick_cleared":
       room.last_trick_cards.clear()
   ```

4. **Include in snapshot** ([`server/game_flow.py:42`](../server/game_flow.py:42)):
   ```python
   "trick_cards": {str(k): v for k, v in room.cur_trick_cards.items()},
   "last_trick_cards": {str(k): v for k, v in room.last_trick_cards.items()},
   ```

5. **Client restoration** ([`static/game_app.js:900-907`](../static/game_app.js:900-907)):
   ```javascript
   // Restore in-progress trick
   if (data.trick_cards) {
       const entries = Object.entries(data.trick_cards);
       trickPlayCount = entries.length;
       for (const [pidxStr, c] of entries) {
           showCardInTrick(parseInt(pidxStr), c);
       }
   }
   // Also restore last completed trick if it hasn't been cleared yet
   else if (data.last_trick_cards) {
       const entries = Object.entries(data.last_trick_cards);
       trickPlayCount = entries.length;
       for (const [pidxStr, c] of entries) {
           showCardInTrick(parseInt(pidxStr), c);
       }
   }
   ```

**Benefits:**
- Clearer separation of concerns (current vs. last trick)
- More explicit about what's being preserved

**Drawbacks:**
- More code changes (server + client)
- Additional state to maintain
- More complex than Option 1

### Option 3: Replay Recent Events on Reconnect

**Change:** Buffer recent game events and replay them to reconnecting clients.

**Rationale:**
- Generalizes to other potential missing-event scenarios
- Clients see the natural event flow rather than a synthetic snapshot

**Implementation:**
- Add `recent_events` ring buffer to room state (last 10-20 events)
- On reconnect, emit buffered events before resuming live stream
- Client processes them as if they arrived in real-time

**Drawbacks:**
- Significantly more complex
- Event replay can cause duplicate processing if not carefully designed
- Overkill for this specific issue

## Recommendation

**Implement Option 1: Preserve Trick Cards Until Cleared**

### Justification

1. **Simplest fix** — 2-line change, no client modifications
2. **Correct semantics** — Server state lifetime matches UI lifetime
3. **Complete fix** — Eliminates the race condition entirely
4. **Low risk** — Minimal memory overhead, no behavior changes for connected clients
5. **Fast to implement** — Can be done in <30 minutes with testing

### Implementation Steps

1. Modify [`server/game_flow.py:389`](../server/game_flow.py:389):
   - Remove `room.cur_trick_cards.clear()` from `trick_won` handler

2. Add `trick_cleared` handler in [`server/game_flow.py`](../server/game_flow.py) (after line 407):
   ```python
   if event == "trick_cleared":
       room.cur_trick_cards.clear()
       socketio.emit(event, send_data, room=room.code)
       return
   ```

3. Test scenarios:
   - Normal play (no disconnection) — verify trick clearing still works
   - Disconnect during trick, reconnect before `trick_won` — verify cards visible
   - Disconnect during trick, reconnect after `trick_won` but before `trick_cleared` — **verify all 4 cards visible**
   - Disconnect during trick, reconnect after `trick_cleared` — verify empty trick area

4. Add regression test in [`tests/test_browser_integration.py`](../tests/test_browser_integration.py):
   - Simulate disconnect mid-trick
   - Wait for trick completion
   - Reconnect during the 1.4s `TRICK_CLEAR_DELAY` window
   - Assert all 4 trick cards are visible in the snapshot

## Alternative: Quick Workaround (If Full Fix Delayed)

If Option 1 can't be implemented immediately, a temporary workaround:

**Client-side:** Don't clear trick area on reconnect if `trickPlayCount > 0` and `trick_cards` is empty:

```javascript
// In game_state_snapshot handler, line 848
if (Object.keys(data.trick_cards || {}).length === 0 && trickPlayCount > 0) {
    // Don't clear — we're in the post-trick-won, pre-trick-cleared window
    // The trick_cleared event will arrive shortly and clear it properly
} else {
    clearTrickArea();
}
```

**Drawback:** Doesn't fix the root cause; cards remain invisible if client was disconnected when they were played.

## Conclusion

The missing UI updates are caused by `cur_trick_cards` being cleared on `trick_won` (when the trick completes) rather than on `trick_cleared` (when the UI clears). This creates a 1.4-second window where reconnecting clients receive an empty snapshot despite cards being visible on other clients' screens.

**Recommended fix:** Move the `cur_trick_cards.clear()` call from the `trick_won` handler to a new `trick_cleared` handler. This aligns server state lifetime with client UI lifetime and completely eliminates the race condition.
