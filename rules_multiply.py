from dataclasses import dataclass
from itertools import combinations
from typing import Any, List
from collections import Counter

# 1) Symbol, Pattern, RuleRow definitions
@dataclass(frozen=True)
class Symbol:
    kind:   str       # 'digit', 'sum', 'carry'
    side:   Any       # 'L', 'R', or None
    pos:    int       # decimal position
    value:  int       # 0–9
    level:  int       # iteration pass

@dataclass(frozen=True)
class Pattern:
    kind:   Any       # exact kind or tuple of kinds or None
    side:   Any       # exact side or None
    pos:    Any       # exact position or None
    value:  Any       # exact value or None
    level:  Any       # exact level or None

    def matches(self, s: Symbol) -> bool:
        return (
            (self.kind  is None or
             (isinstance(self.kind, tuple) and s.kind in self.kind) or
              s.kind  == self.kind)
        and (self.side  is None or s.side  == self.side)
        and (self.pos   is None or s.pos   == self.pos)
        and (self.value is None or s.value == self.value)
        and (self.level is None or s.level == self.level)
        )

@dataclass(frozen=True)
class RuleRow:
    p1:          Pattern
    p2:          Pattern
    output:      Symbol
    commutative: bool

# 2) apply_rules: pure pattern‐matching
def apply_rules(symbols: List[Symbol], rules: List[RuleRow]) -> List[Symbol]:
    used = set()
    out  = []
    for i, j in combinations(range(len(symbols)), 2):
        if i in used or j in used:
            continue
        s1, s2 = symbols[i], symbols[j]
        for rule in rules:
            # forward match
            if rule.p1.matches(s1) and rule.p2.matches(s2):
                out.append(rule.output)
                used |= {i, j}
                break
            # swapped if commutative
            if rule.commutative and rule.p1.matches(s2) and rule.p2.matches(s1):
                out.append(rule.output)
                used |= {i, j}
                break
    # propagate unused
    for idx, s in enumerate(symbols):
        if idx not in used:
            out.append(s)
    return out

def multiset(symbols: List[Symbol]) -> Counter:
    return Counter((s.kind, s.side, s.pos, s.value, s.level) for s in symbols)

# 3) Build the full rule set for given operand lengths
def build_rules(m: int, n: int) -> List[RuleRow]:
    rules: List[RuleRow] = []

    # 3a) Multiplication rules (level 1)
    for i in range(m):
        for j in range(n):
            for d1 in range(10):
                for d2 in range(10):
                    lo = (d1 * d2) % 10
                    hi = (d1 * d2) // 10
                    # low digit
                    rules.append(RuleRow(
                        p1=Pattern('digit','L', i, d1, 0),
                        p2=Pattern('digit','R', j, d2, 0),
                        output=Symbol('sum', None,   i + j,   lo, 1),
                        commutative=False
                    ))
                    # carry digit
                    rules.append(RuleRow(
                        p1=Pattern('digit','L', i, d1, 0),
                        p2=Pattern('digit','R', j, d2, 0),
                        output=Symbol('carry', None, i + j+1, hi, 1),
                        commutative=False
                    ))

    # 3b) Addition rules (level 2), exploit commutativity: only v2>=v1
    max_pos = m + n
    for p in range(max_pos + 1):
        for v1 in range(10):
            for v2 in range(v1, 10):
                slo = (v1 + v2) % 10
                shi = (v1 + v2) // 10
                rules.append(RuleRow(
                    p1=Pattern(('sum','carry'), None, p, v1, 1),
                    p2=Pattern(('sum','carry'), None, p, v2, 1),
                    output=Symbol('sum', None, p, slo, 2),
                    commutative=True
                ))
                rules.append(RuleRow(
                    p1=Pattern(('sum','carry'), None, p, v1, 1),
                    p2=Pattern(('sum','carry'), None, p, v2, 1),
                    output=Symbol('carry', None, p+1, shi, 2),
                    commutative=True
                ))

    # 3c) Carry‐add rules (level 3)
    max_pos_c = m + n + 1
    for p in range(max_pos_c + 1):
        for s in range(10):
            for c in (0,1):
                dlo = (s + c) % 10
                dhi = (s + c) // 10
                # sum after carry
                rules.append(RuleRow(
                    p1=Pattern('sum',   None, p,   s, 2),
                    p2=Pattern('carry', None, p,   c, 2),
                    output=Symbol('sum',   None, p,   dlo, 3),
                    commutative=False
                ))
                # carry after carry
                rules.append(RuleRow(
                    p1=Pattern('sum',   None, p,   s, 2),
                    p2=Pattern('carry', None, p,   c, 2),
                    output=Symbol('carry', None, p+1, dhi, 3),
                    commutative=False
                ))

    return rules

# 4) The only algorithm: fixed‐point multiplication
def multiply(A: int, B: int) -> int:
    A_digits = list(map(int, str(A)[::-1]))
    B_digits = list(map(int, str(B)[::-1]))
    m, n     = len(A_digits), len(B_digits)

    import ipdb; ipdb.set_trace()
    symbols = [
        Symbol('digit','L', i, d, 0)
        for i, d in enumerate(A_digits)
    ] + [
        Symbol('digit','R', j, d, 0)
        for j, d in enumerate(B_digits)
    ]

    rules = build_rules(m, n)
    seen  = multiset(symbols)

    while True:
        nxt = apply_rules(symbols, rules)
        cur = multiset(nxt)
        if cur == seen:
            break
        symbols, seen = nxt, cur

    return sum(s.value * (10**s.pos) for s in symbols if s.kind in ('sum','carry'))

# 5) Test suite
if __name__ == '__main__':
    # for a, b in [(123, 45), (999, 999), (1024, 2048), (4, 5), (0, 12345)]:
        # assert multiply(a, b) == a * b, f"{a}*{b} => {multiply(a, b)}"

    for a, b in [(4, 5)]:
        assert multiply(a, b) == a * b, f"{a}*{b} => {multiply(a, b)}"
    print("All tests passed!")
