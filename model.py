"""Texas hold'em bot.

Strategy:
  * Preflop: hand equity vs a random hand, learned per hand class by Monte
    Carlo and cached; mixed raise/call/fold frequencies by situation
    (open, defend, 3-bet, 4-bet/shove).
  * Postflop: Monte Carlo equity against an estimated opponent range that
    tightens with bet size and street; board texture and range advantage
    shape value bets, c-bets, semi-bluffs and bluffs.
  * Opponent model: per-team VPIP, PFR, aggression and fold-to-bet, used to
    classify the opponent (nit, station, maniac, fit-or-fold, reg) and
    adjust ranges, bluff frequency and calling margins.
  * Adaptive simulation: stops early when the decision is clear and spends
    the time budget on close calls.
  * All state is plain data, so the model pickles; `run` falls back to a
    legal action if anything goes wrong.
"""

import math
import random
import time

from card_utils import evaluate
from player_action import PlayerAction

_DECK = None
_RNG = random.Random(0xBEEF)   # module-level so the pickled model holds plain data only


def _deck():
    global _DECK
    if _DECK is None:
        from card_utils import full_deck
        _DECK = list(full_deck())
    return _DECK


_SUITED_BONUS = 0.035
_OFFSUIT_GAP_PENALTY = {1: 0.0, 2: -0.005, 3: -0.015, 4: -0.03,
                        5: -0.045, 6: -0.055, 7: -0.06, 8: -0.065,
                        9: -0.07, 10: -0.075, 11: -0.08, 12: -0.085}


def _prior_eq(a, b):
    hi = max(a.value, b.value)
    lo = min(a.value, b.value)
    if hi == lo:
        return 0.50 + 0.0292 * (hi - 2)
    e = 0.285 + 0.0165 * (hi - 2) + 0.0125 * (lo - 2)
    if a.suit == b.suit:
        e += _SUITED_BONUS
    gap = hi - lo
    if gap in _OFFSUIT_GAP_PENALTY:
        e += _OFFSUIT_GAP_PENALTY[gap]
    return min(0.85, max(0.30, e))


def _hand_key(a, b):
    hi = max(a.value, b.value)
    lo = min(a.value, b.value)
    return "%d-%d-%d" % (hi, lo, 1 if a.suit == b.suit else 0)


def _texture(board):
    """Return (dry, wet, paired, monotone, ace_high, broadway)."""
    if not board:
        return (False, False, False, False, False, False)
    values = sorted(c.value for c in board)
    suits = [c.suit for c in board]
    paired = len(set(values)) < len(values)
    suits_set = set(suits)
    monotone = len(board) >= 3 and len(suits_set) == 1
    if len(suits_set) == 2 and len(board) >= 3:
        common = max(suits.count(s) for s in suits_set)
        monotone = common >= 3 and len(board) >= 3
    ace_high = 14 in values
    broadway = sum(1 for v in values if v >= 10) >= 2
    connected = (values[-1] - values[0]) <= 4 and len(set(values)) == len(values)
    dry = (not paired) and (not monotone) and (not connected) and not broadway
    wet = connected or monotone or (paired and broadway)
    return (dry, wet, paired, monotone, ace_high, broadway)


# ----------------------------------------------------------------------
# Preflop mixes. `open_adj` shifts our opening range (negative = wider).
# `defend_adj` shifts our defending range (negative = wider).
# ----------------------------------------------------------------------
def _mix_open_btn(e, open_adj):
    if e >= 0.63 + open_adj:
        return (1.0, 0.0)
    if e >= 0.52 + open_adj:
        return (0.88, 0.12)
    if e >= 0.44 + open_adj:
        return (0.55, 0.35)
    if e >= 0.38 + open_adj:
        return (0.20, 0.45)
    return (0.0, 0.05)


def _mix_open_bb(e, open_adj):
    if e >= 0.60 + open_adj:
        return (1.0, 0.0)
    if e >= 0.50 + open_adj:
        return (0.55, 0.45)
    return (0.0, 1.0)


def _mix_vs_open(e, defend_adj):
    if e >= 0.62 + defend_adj:
        return (0.75, 0.25)
    if e >= 0.52 + defend_adj * 0.6:
        return (0.20, 0.70)
    if e >= 0.45 + defend_adj:
        return (0.0, 0.85)
    return (0.0, 0.0)


def _mix_vs_3bet(e, defend_adj):
    if e >= 0.68 + defend_adj:
        return (0.85, 0.15)
    if e >= 0.57 + defend_adj * 0.6:
        return (0.10, 0.80)
    if e >= 0.50 + defend_adj * 0.4:
        return (0.0, 0.55)
    return (0.0, 0.0)


def _mix_vs_4bet(e, defend_adj):
    if e >= 0.70 + defend_adj:
        return (1.0, 0.0)
    if e >= 0.62 + defend_adj * 0.6:
        return (0.55, 0.45)
    if e >= 0.55 + defend_adj * 0.4:
        return (0.0, 0.60)
    return (0.0, 0.0)


# ----------------------------------------------------------------------
# Opponent typing (HU-relative). Exploit dict keys:
#   open_adj, defend_adj, cbet_adj, bluff_adj, facing_bet_margin
# ----------------------------------------------------------------------
def _opponent_type(st):
    if st["hands"] < 8 or st["acts"] < 20:
        return ("unknown", {})
    vpip = st["vpip"] / max(1, st["hands"])
    pfr = st["pfr"] / max(1, st["hands"])
    aggr = st["raise"] / max(1, st["acts"])
    ftb = (st["ftb"] + 2.0) / (st["faced"] + 5.0)

    # Tight HU bots (VPIP < ~52%, PFR < ~43%) are the 'tight' dummies.
    if vpip < 0.52 and pfr < 0.43:
        return ("nit", {
            "open_adj": -0.05,
            "defend_adj": +0.04,
            "cbet_adj": +0.40,
            "bluff_adj": +0.18,
            "facing_bet_margin": +0.03,
        })
    if vpip > 0.82 and pfr < 0.40:
        return ("station", {
            "open_adj": -0.02,
            "defend_adj": -0.03,
            "cbet_adj": +0.10,
            "bluff_adj": -0.20,
            "facing_bet_margin": -0.02,
        })
    if pfr > 0.65 or aggr > 0.60:
        return ("maniac", {
            "open_adj": +0.03,
            "defend_adj": -0.04,
            "cbet_adj": -0.10,
            "bluff_adj": -0.15,
            "facing_bet_margin": -0.03,
        })
    if ftb > 0.70 and st["faced"] >= 5:
        return ("fit_or_fold", {
            "open_adj": -0.03,
            "defend_adj": -0.02,
            "cbet_adj": +0.25,
            "bluff_adj": +0.22,
            "facing_bet_margin": 0.0,
        })
    return ("reg", {
        "open_adj": 0.0, "defend_adj": 0.0,
        "cbet_adj": 0.0, "bluff_adj": 0.0,
        "facing_bet_margin": 0.0,
    })


class Model:
    TIME_BUDGET = 0.30       # seconds of Monte Carlo per decision
    MIN_ITERS = 60
    MAX_ITERS = 2000
    CHUNK = 30
    PRE_CACHE_N = 4000

    @property
    def rng(self):
        return _RNG

    def __init__(self):
        self.hand = []
        self.player_id = None
        self.pending_bet = False
        self.hands_seen = 0
        self.opp = {}
        self.pre_cache = {}
        self.seen_voluntary = set()   # per-hand VPIP / PFR gate
        _deck()                       # warm the deck cache

    def seed(self, hand, player_id):
        self.hand = list(hand)
        self.player_id = player_id
        self.pending_bet = False
        self.seen_voluntary = set()

    def end_game(self, actions, game_state):
        self.hands_seen += 1
        try:
            # The opponent's last action (e.g. a fold) is only seen here.
            self._observe(actions, game_state, self.player_id)
            for p in range(game_state.num_players):
                if p != self.player_id:
                    self._stats(game_state.teams[p])["hands"] += 1
        except Exception:
            pass

    def run(self, actions, game_state):
        gs = game_state
        me = self.player_id
        act = None
        try:
            self._observe(actions, gs, me)
            if not gs.board:
                act = self._preflop(gs, me)
            else:
                act = self._postflop(gs, me)
        except Exception:
            act = None
        if act is None:
            try:
                act = (PlayerAction.call(me) if gs.call_amount == 0
                       else PlayerAction.fold(me))
            except Exception:
                act = PlayerAction.fold(me)
        if getattr(act, "kind", "") == "raise":
            self.pending_bet = True
        return act

    # ------------------------------------------------------------------
    # Opponent model
    # ------------------------------------------------------------------
    def _stats(self, team):
        st = self.opp.get(team)
        if st is None:
            st = {"hands": 0, "acts": 0, "raise": 0, "fold": 0, "call": 0,
                  "faced": 0, "ftb": 0, "rtb": 0, "vpip": 0, "pfr": 0}
            self.opp[team] = st
        return st

    def _observe(self, actions, gs, me):
        teams = gs.teams
        preflop = len(gs.board) == 0
        for a in actions:
            pid = a.player_id
            if pid is None or pid < 0 or pid == me:
                continue
            if a.kind not in ("fold", "call", "raise"):
                continue
            st = self._stats(teams[pid])
            st["acts"] += 1
            if a.kind == "raise":
                st["raise"] += 1
                if preflop and pid not in self.seen_voluntary:
                    st["pfr"] += 1
            elif a.kind == "call":
                st["call"] += 1
            elif a.kind == "fold":
                st["fold"] += 1
            # VPIP / PFR are once-per-hand, preflop-only.
            if (preflop and a.kind in ("call", "raise")
                    and pid not in self.seen_voluntary):
                st["vpip"] += 1
                self.seen_voluntary.add(pid)
            if self.pending_bet:
                self.pending_bet = False
                st["faced"] += 1
                if a.kind == "fold":
                    st["ftb"] += 1
                elif a.kind == "raise":
                    st["rtb"] += 1

    def _profile(self, gs, me):
        opps = [p for p in range(gs.num_players) if p != me]
        if not opps:
            return ("unknown", {}, 0.4, 0.2)
        st = self.opp.get(gs.teams[opps[0]])
        if st is None:
            return ("unknown", {}, 0.4, 0.2)
        typ, exploit = _opponent_type(st)
        fb = (st["ftb"] + 2.0) / (st["faced"] + 5.0)
        rr = (st["raise"] + 1.0) / (st["acts"] + 5.0)
        return (typ, exploit, fb, rr)

    # ------------------------------------------------------------------
    # Preflop equity cache
    # ------------------------------------------------------------------
    def _pre_equity(self):
        a, b = self.hand
        key = _hand_key(a, b)
        rec = self.pre_cache.get(key)
        if rec is None:
            rec = [0.0, 0]
            self.pre_cache[key] = rec
        if rec[1] < self.PRE_CACHE_N:
            known = {a, b}
            deck = [c for c in _deck() if c not in known]
            rng = self.rng
            t0 = time.perf_counter()
            for _ in range(120):
                s = rng.sample(deck, 7)
                board = s[2:]
                try:
                    mine = evaluate([a, b] + board)
                    theirs = evaluate([s[0], s[1]] + board)
                except ValueError:
                    continue
                rec[1] += 1
                if mine > theirs:
                    rec[0] += 1.0
                elif mine == theirs:
                    rec[0] += 0.5
                if time.perf_counter() - t0 > 0.008:
                    break
        w = 25.0
        return (rec[0] + _prior_eq(a, b) * w) / (rec[1] + w)

    # ------------------------------------------------------------------
    # Sizing helpers
    # ------------------------------------------------------------------
    def _raise_to(self, gs, amount):
        lo, hi = gs.raise_range
        return PlayerAction.raise_to(self.player_id,
                                     int(max(lo, min(hi, amount))))

    def _raise_pot(self, gs, frac):
        lo, hi = gs.raise_range
        pot = gs.pot
        call = gs.call_amount
        target = gs.current_bet + frac * (pot + call)
        amt = int(max(lo, min(hi, target)))
        if (hi - amt) < 0.55 * (hi - gs.bets[self.player_id]):
            amt = int(hi)
        return PlayerAction.raise_to(self.player_id, amt)

    # ------------------------------------------------------------------
    # PREFLOP
    # ------------------------------------------------------------------
    def _preflop(self, gs, me):
        bb = gs.big_blind
        call = gs.call_amount
        po = gs.pot_odds
        cb = gs.current_bet / float(bb)
        can_raise = bool(gs.can_raise) and "raise" in gs.legal_actions
        typ, exploit, fb, rr = self._profile(gs, me)
        e = self._pre_equity()
        r = self.rng.random()
        open_adj = exploit.get("open_adj", 0.0)
        defend_adj = exploit.get("defend_adj", 0.0)

        # Maniac: pure equity, ignore range tricks.
        if typ == "maniac":
            if call == 0:
                return PlayerAction.call(me)
            threshold = po + 0.02 if cb <= 1.0 else po + 0.05
            return (PlayerAction.call(me) if e >= threshold
                    else PlayerAction.fold(me))

        # Facing 4bet / shove
        if cb > 12.0:
            raise_p, call_p = _mix_vs_4bet(e, defend_adj)
            if can_raise and r < raise_p:
                return self._raise_to(gs, gs.raise_range[1])
            if r < raise_p + call_p:
                return PlayerAction.call(me)
            return PlayerAction.fold(me)

        # Facing 3bet
        if cb > 4.5:
            raise_p, call_p = _mix_vs_3bet(e, defend_adj)
            if can_raise and r < raise_p:
                return self._raise_to(gs, min(gs.raise_range[1],
                                              int(cb * bb * 2.4)))
            if r < raise_p + call_p:
                return PlayerAction.call(me)
            return PlayerAction.fold(me)

        # Facing an open
        if cb > 1.0 + 1e-9:
            raise_p, call_p = _mix_vs_open(e, defend_adj)
            if can_raise and r < raise_p:
                return self._raise_to(gs, min(gs.raise_range[1],
                                              int(cb * bb * 3.0)))
            if r < raise_p + call_p:
                return PlayerAction.call(me)
            return PlayerAction.fold(me)

        # Unopened: button (SB, HU) or BB option.
        if gs.bets[me] == gs.small_blind and gs.button == me:
            raise_p, call_p = _mix_open_btn(e, open_adj)
            if can_raise and r < raise_p:
                return self._raise_to(gs, int(2.2 * bb))
            return PlayerAction.call(me)

        raise_p, call_p = _mix_open_bb(e, open_adj)
        if can_raise and r < raise_p:
            return self._raise_to(gs, int(3.0 * bb))
        return PlayerAction.call(me)

    # ------------------------------------------------------------------
    # POSTFLOP
    # ------------------------------------------------------------------
    def _postflop(self, gs, me):
        board = list(gs.board)
        call = gs.call_amount
        po = gs.pot_odds
        can_raise = bool(gs.can_raise) and "raise" in gs.legal_actions
        typ, exploit, fb, rr = self._profile(gs, me)
        dry, wet, paired, mono, ace_high, broadway = _texture(board)
        r = self.rng.random()

        opp_range_lo = self._range_lo(gs, typ)
        if call > 0 and typ not in ("maniac", "station"):
            # A bigger bet means a stronger range.
            f = call / max(1.0, gs.pot - call)
            bump = 0.0 if f < 0.3 else 0.05 if f < 0.6 else 0.09 if f < 1.0 else 0.13
            bump += 0.015 * (len(board) - 3)
            opp_range_lo = min(0.62, opp_range_lo + bump)
        if call > 0:
            targets = (po, 0.66, 0.78)
        else:
            targets = (0.52, 0.66, 0.80)
        eq = self._equity(board, opp_range_lo, targets)
        adv = self._range_advantage(gs, me, typ, dry, wet, paired, mono,
                                    ace_high, broadway)
        draw = self._has_draw(self.hand, board)

        if call > 0:
            return self._decide_facing_bet(
                gs, me, eq, po, can_raise, draw, typ, exploit, adv, r)
        return self._decide_unchecked(
            gs, me, eq, can_raise, draw, typ, exploit, adv, fb, r)

    def _range_lo(self, gs, typ):
        if typ == "nit":
            return 0.50
        if typ == "maniac":
            return 0.30
        if typ == "station":
            return 0.32
        if typ == "fit_or_fold":
            return 0.40
        return 0.38

    def _range_advantage(self, gs, me, typ, dry, wet, paired, mono,
                         ace_high, broadway):
        adv = 0.0
        if ace_high:
            adv += 0.25
        if not paired and not mono and not wet:
            adv += 0.15
        if mono:
            adv -= 0.20
        if paired:
            adv -= 0.10
        if broadway and not ace_high:
            adv -= 0.05
        if typ == "nit":
            adv += 0.10
        if typ == "station":
            adv -= 0.10
        return max(-1.0, min(1.0, adv))

    def _decide_facing_bet(self, gs, me, eq, po, can_raise, draw,
                           typ, exploit, adv, r):
        bluff_adj = exploit.get("bluff_adj", 0.0)
        margin_adj = exploit.get("facing_bet_margin", 0.0)

        # Strong value
        if can_raise and eq >= 0.78:
            return self._raise_pot(gs, 0.85)
        if can_raise and eq >= 0.66 and r < 0.55:
            return self._raise_pot(gs, 0.70)

        # Semi-bluff with draws
        if can_raise and draw and eq < po + 0.06:
            bluff_p = 0.20 + 0.15 * adv + bluff_adj * 0.5
            if typ == "station" or typ == "maniac":
                bluff_p = 0.0
            if r < max(0.0, min(0.55, bluff_p)):
                return self._raise_pot(gs, 0.75)

        # Station: never bluff, call wider (their bets are weak)
        if typ == "station":
            return (PlayerAction.call(me) if eq >= po + 0.02
                    else PlayerAction.fold(me))

        # Maniac: call wider
        if typ == "maniac":
            return (PlayerAction.call(me) if eq >= po + 0.01
                    else PlayerAction.fold(me))

        # Default: pot-odds call with type-aware margin
        margin = 0.02 + margin_adj
        if adv > 0.3:
            margin = max(0.005, margin - 0.02)
        if adv < -0.3:
            margin += 0.03
        return (PlayerAction.call(me) if eq >= po + margin
                else PlayerAction.fold(me))

    def _decide_unchecked(self, gs, me, eq, can_raise, draw,
                          typ, exploit, adv, fb, r):
        if not can_raise:
            return PlayerAction.call(me)

        cbet_adj = exploit.get("cbet_adj", 0.0)
        bluff_adj = exploit.get("bluff_adj", 0.0)

        # Strong value
        if eq >= 0.80:
            return self._raise_pot(gs, 0.75)
        if eq >= 0.66:
            size = 0.85 if typ == "station" else 0.60
            return self._raise_pot(gs, size)

        # C-bet / thin value
        if eq >= 0.52:
            if typ == "station":
                p = 0.75
            elif typ == "maniac":
                p = 0.30
            elif fb > 0.55:
                p = 0.55
            else:
                p = 0.30 + cbet_adj
            if r < max(0.0, min(0.85, p)):
                return self._raise_pot(gs, 0.55)

        # Semi-bluff draws
        if draw and eq < 0.58:
            p = 0.30 + 0.15 * adv + bluff_adj
            if typ == "station":
                p = 0.05
            if typ == "maniac":
                p = 0.10
            if r < max(0.0, min(0.75, p)):
                return self._raise_pot(gs, 0.60)

        # Pure bluff
        if eq < 0.45:
            p = 0.05 + bluff_adj
            if adv > 0.15:
                p += 0.10
            if typ == "station" or typ == "maniac":
                p = 0.0
            if r < max(0.0, min(0.60, p)):
                return self._raise_pot(gs, 0.55)

        return PlayerAction.call(me)

    # ------------------------------------------------------------------
    # Monte Carlo vs an estimated range
    # ------------------------------------------------------------------
    def _equity(self, board, opp_range_lo, targets=(0.5,)):
        hand = self.hand
        known = set(hand) | set(board)
        deck = [c for c in _deck() if c not in known]
        need = 5 - len(board)
        fixed = evaluate(hand + board) if need == 0 else None
        rng = self.rng
        wins = ties = n = 0
        eq = 0.5
        t0 = time.perf_counter()
        while n < self.MAX_ITERS:
            for _ in range(self.CHUNK):
                opp = None
                for _try in range(10):
                    s = rng.sample(deck, 2 + need)
                    if opp_range_lo <= 0.0 or _prior_eq(s[0], s[1]) >= opp_range_lo:
                        opp = s
                        break
                if opp is None:
                    opp = rng.sample(deck, 2 + need)
                sim = board + opp[2:]
                try:
                    mine = fixed if fixed is not None else evaluate(hand + sim)
                    theirs = evaluate([opp[0], opp[1]] + sim)
                except ValueError:
                    continue
                n += 1
                if mine > theirs:
                    wins += 1
                elif mine == theirs:
                    ties += 1
            if n == 0:
                break
            eq = (wins + 0.5 * ties) / n
            se = math.sqrt(max(eq * (1.0 - eq), 0.05) / n)
            if n >= self.MIN_ITERS and all(abs(eq - t) > 2.5 * se for t in targets):
                break
            if time.perf_counter() - t0 > self.TIME_BUDGET:
                break
        return eq

    @staticmethod
    def _has_draw(hand, board):
        if len(board) >= 5:
            return False
        try:
            if evaluate(list(hand) + list(board))[0] >= 3:
                return False
        except ValueError:
            return False
        cards = list(hand) + list(board)
        for s in "CDHS":
            total = sum(1 for c in cards if c.suit == s)
            mine = sum(1 for c in hand if c.suit == s)
            if total == 4 and mine >= 1:
                return True
        vals = {c.value for c in cards}
        hv = {c.value for c in hand}
        if 14 in vals:
            vals.add(1)
        if 14 in hv:
            hv.add(1)
        for start in range(1, 11):
            w = set(range(start, start + 5))
            if len(w & vals) == 4 and (w & hv):
                return True
        return False
