import json
import os
import re
import sys

SOURCE_DIR = "raw_data"
RESULT_DIR = "knowledge_base"
TERMS_MAP_PATH = "terms_map.json"
GENERIC_FIRST = (
    "битв", "войн", "орден", "приказ", "галактич", "имперск",
    "повстанч", "звёздн", "тысячелет", "тёмн", "светов",
    "генерал", "граф", "дарт", "звезд",
)
SHORT_LONE = ("люк", "лея", "хан", "бен")
SHORT_TAILS = "аеоыиу"
LONG_TAILS = ("ом", "ем", "ам", "ах", "ой", "ей", "ым", "их", "ю")


def load_terms_map(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if all(isinstance(value, dict) for value in data.values()):
        return {orig: repl for category in data.values()
                for orig, repl in category.items()}
    return data


def normalize(word):
    return word.casefold().replace("ё", "е").replace("ль", "л")


def stem_match(segment, keyword):
    seg = normalize(segment)
    kw = normalize(keyword)
    if seg == kw:
        return True, 'exact'

    n, m = len(kw), len(seg)
    common = 0
    for a, b in zip(seg, kw):
        if a != b:
            break
        common += 1

    if common == n and m > n:
        return True, 'derivative'

    if n <= 3:
        if m == n:
            ok = common >= n - 1
            return ok, ('exact' if ok else None)
        tail = seg[n:]
        if m == n + 1:
            ok = common == n and tail in SHORT_TAILS
            return ok, ('exact' if ok else None)
        if m == n + 2:
            ok = common == n and tail in LONG_TAILS
            return ok, ('exact' if ok else None)
        return False, None

    if m > n + 2:
        return False, None

    ok = common >= max(4, n - 2)
    return ok, ('exact' if ok else None)


def make_replacement(original_word, keyword, repl):
    first_char = original_word[0]
    is_upper = first_char.isupper()

    seg_norm = normalize(original_word)
    kw_norm = normalize(keyword)
    n = len(kw_norm)

    if seg_norm == kw_norm:
        return repl if is_upper else repl.lower()

    if seg_norm.startswith(kw_norm) and len(seg_norm) > n:
        suffix_norm = seg_norm[n:]
        suffix = original_word[len(original_word) - len(suffix_norm):]
        base = repl if is_upper else repl.lower()
        return base + suffix

    return repl if is_upper else repl.lower()


def parts_of(term):
    return re.findall(r"[а-яёa-z0-9]+", term, re.IGNORECASE)


def build_rules(terms_map):
    entries = []
    for term, repl in terms_map.items():
        parts = parts_of(term)
        if parts:
            entries.append((parts, repl))
    entries.sort(key=lambda pair: (len(pair[0]), sum(len(w) for w in pair[0])),
                 reverse=True)

    first_counts = {}
    for parts, _ in entries:
        first = parts[0].casefold()
        first_counts[first] = first_counts.get(first, 0) + 1

    lone = []
    for parts, repl in entries:
        if len(parts) < 2:
            continue
        first = parts[0].casefold()
        if first_counts[first] != 1:
            continue
        if first.startswith(GENERIC_FIRST):
            continue
        if len(first) < 4 and first not in SHORT_LONE:
            continue
        lone.append((parts[0], repl))

    return entries, lone


def apply_replacements(text, entries, lone):
    segs = list(re.finditer(r"[а-яёa-z0-9]+", text, re.IGNORECASE))
    used = [False] * len(segs)

    replacements = []

    for parts, repl in entries:
        size = len(parts)
        i = 0
        while i < len(segs):
            if used[i]:
                i += 1
                continue

            matched, match_type = stem_match(segs[i].group(0), parts[0])
            if not matched:
                i += 1
                continue

            if size == 1:
                start_pos = segs[i].start()
                end_pos = segs[i].end()
                replaced = make_replacement(segs[i].group(0), parts[0], repl)
                replacements.append((start_pos, end_pos, replaced))
                used[i] = True
                i += 1
                continue

            j = i + 1
            all_match = True
            while j < len(segs) and j - i < size:
                if used[j]:
                    all_match = False
                    break
                matched_j, _ = stem_match(segs[j].group(0), parts[j - i])
                if not matched_j:
                    all_match = False
                    break
                j += 1

            matched_count = j - i
            if all_match and matched_count == size:
                start_pos = segs[i].start()
                end_pos = segs[j - 1].end()
                first_char = segs[i].group(0)[0]
                replaced = repl if first_char.isupper() else repl.lower()
                replacements.append((start_pos, end_pos, replaced))
                for k in range(i, j):
                    used[k] = True
                i = j
            else:
                i += 1

    for first, repl in lone:
        for i, seg in enumerate(segs):
            if not used[i]:
                matched, _ = stem_match(seg.group(0), first)
                if matched:
                    start_pos = seg.start()
                    end_pos = seg.end()
                    replaced = make_replacement(seg.group(0), first, repl)
                    replacements.append((start_pos, end_pos, replaced))
                    used[i] = True

    replacements.sort(key=lambda x: x[0])

    filtered = []
    last_end = 0
    for start, end, repl in replacements:
        if start >= last_end:
            filtered.append((start, end, repl))
            last_end = end
    replacements = filtered

    result = []
    pos = 0
    for start, end, repl in replacements:
        result.append(text[pos:start])
        result.append(repl)
        pos = end
    result.append(text[pos:])

    return "".join(result)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    entries, lone = build_rules(load_terms_map(TERMS_MAP_PATH))
    os.makedirs(RESULT_DIR, exist_ok=True)

    files = sorted(name for name in os.listdir(SOURCE_DIR)
                   if name.endswith(".md"))
    if argv:
        wanted = {name for name in files if name.endswith(".md")
                  and name in argv}
        files = [name for name in files if name in wanted]
    for fname in files:
        with open(os.path.join(SOURCE_DIR, fname), encoding="utf-8") as fh:
            text = fh.read()

        text = apply_replacements(text, entries, lone)

        with open(os.path.join(RESULT_DIR, fname), "w", encoding="utf-8") as fh:
            fh.write(text)

if __name__ == "__main__":
    sys.exit(main())
