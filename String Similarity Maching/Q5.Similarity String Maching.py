import difflib

def align_strings(s1: str, s2: str):
    """
    Align two strings using a simple algorithm (via difflib) and return aligned versions
    with “-” gaps inserted.  
    This is a heuristic alignment, not optimal like Needleman‑Wunsch, but sufficient for short strings.
    """
    # We use SequenceMatcher to get matching blocks and build aligned strings
    matcher = difflib.SequenceMatcher(None, s1, s2)
    aligned1 = []
    aligned2 = []
    i1 = i2 = 0

    for block in matcher.get_matching_blocks():
        (a1, a2, length) = block
        # fill unmatched leading region
        # region in s1 from i1 to a1 aligns to region in s2 from i2 to a2
        while i1 < a1 or i2 < a2:
            if (a1 - i1) > (a2 - i2):
                # more chars in s1 side → insert gap in s2
                aligned1.append(s1[i1])
                aligned2.append('-')
                i1 += 1
            elif (a2 - i2) > (a1 - i1):
                # more chars in s2 side → insert gap in s1
                aligned1.append('-')
                aligned2.append(s2[i2])
                i2 += 1
            else:
                # same length: align them directly (mismatches)
                aligned1.append(s1[i1])
                aligned2.append(s2[i2])
                i1 += 1
                i2 += 1

        # now the matched block
        for k in range(length):
            aligned1.append(s1[a1 + k])
            aligned2.append(s2[a2 + k])
        i1 = a1 + length
        i2 = a2 + length

    return ''.join(aligned1), ''.join(aligned2)

def compute_similarity(al1: str, al2: str):
    """
    Given two aligned strings (same length, with gaps), compute percentage similarity:
    (number of matching positions) / (number of non-gap positions in longer) * 100
    """
    assert len(al1) == len(al2)
    matches = 0
    total = 0
    for c1, c2 in zip(al1, al2):
        # skip positions where one side has a gap
        if c1 == '-' or c2 == '-':
            continue
        total += 1
        if c1 == c2:
            matches += 1
    if total == 0:
        return 0.0
    return 100.0 * matches / total

def generate_report(al1: str, al2: str):
    """
    Return a report showing matching vs non-matching positions.
    E.g.:

    Aligned1: A B C D - E F
    Aligned2: A B X D G E F
                | |   |   | |

    Report:
     Pos 1: ‘A’ vs ‘A’ → match
     Pos 2: ‘B’ vs ‘B’ → match
     Pos 3: ‘C’ vs ‘X’ → mismatch
     ...
    """
    lines = []
    header1 = "Aligned1: " + ' '.join(al1)
    header2 = "Aligned2: " + ' '.join(al2)
    lines.append(header1)
    lines.append(header2)
    # Mark matches with '|' or mismatch with space
    marker = []
    for c1, c2 in zip(al1, al2):
        if c1 == c2 and c1 != '-':
            marker.append('|')
        else:
            marker.append(' ')
    lines.append("           " + ' '.join(marker))
    lines.append("")
    # Detailed per-position
    for idx, (c1, c2, m) in enumerate(zip(al1, al2, marker)):
        pos = idx + 1
        if c1 == '-' or c2 == '-':
            lines.append(f"Pos {pos}: '{c1}' vs '{c2}' → (gap / alignment) ")
        elif c1 == c2:
            lines.append(f"Pos {pos}: '{c1}' vs '{c2}' → MATCH")
        else:
            lines.append(f"Pos {pos}: '{c1}' vs '{c2}' → MISMATCH")
    return '\n'.join(lines)

def compare_strings(s1: str, s2: str):
    # optional: validate lengths between 6 and 10
    if not (6 <= len(s1) <= 10 and 6 <= len(s2) <= 10):
        raise ValueError("Strings must be 6 to 10 characters long")
    al1, al2 = align_strings(s1, s2)
    sim_percentage = compute_similarity(al1, al2)
    report = generate_report(al1, al2)
    return sim_percentage, report

if __name__ == "__main__":
    s1 = input("Enter first string (6–10 chars): ").strip()
    s2 = input("Enter second string (6–10 chars): ").strip()

    try:
        sim, rpt = compare_strings(s1, s2)
        print(f"Similarity: {sim:.2f}%")
        print("Match Report:")
        print(rpt)
    except ValueError as e:
        print("Error:", e)

