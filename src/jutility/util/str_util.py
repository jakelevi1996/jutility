import textwrap

def remove_duplicate_substring(s: str, sub_str):
    duplicates = sub_str * 2
    while duplicates in s:
        s = s.replace(duplicates, sub_str)

    return s

def clean_string(s: str, allowed_non_alnum_chars="-_.,", replacement="_"):
    s_clean = "".join(
        c if (c.isalnum() or c in allowed_non_alnum_chars) else replacement
        for c in str(s)
    )
    s_clean = remove_duplicate_substring(s_clean, replacement)
    return s_clean

def trim_string(s: str, max_len, suffix="_..._"):
    if len(s) > max_len:
        trim_len = max(max_len - len(suffix), 0)
        s = s[:trim_len] + suffix

    return s

def wrap_string(s: str, max_len=80, wrap_len=60):
    if len(s) > max_len:
        s = textwrap.fill(s, width=wrap_len, break_long_words=False)
    return s

def indent(input_str: str, num_spaces=4):
    return textwrap.indent(input_str, " " * num_spaces)

def extract_substring(s, prefix, suffix, offset=None, strip=True):
    s = str(s)
    start_ind   = s.index(prefix, offset) + len(prefix)
    end_ind     = s.index(suffix, start_ind)
    s_substring = s[start_ind:end_ind]
    if strip:
        s_substring = s_substring.strip()

    return s_substring

def merge_strings(input_list: list[str], clean=True):
    output_str = ""
    while len(input_list) > 0:
        next_char_set = set(s[0] for s in input_list)
        if len(next_char_set) == 1:
            [c] = next_char_set
            output_str += c
            input_list = [s[1:] for s in input_list]
        else:
            remaining_chars = set(c for s in input_list for c in s)
            valid_next_chars = sorted(
                c for c in remaining_chars
                if all((c in s) for s in input_list)
            )
            if len(valid_next_chars) > 0:
                max_ind_dict = {
                    c: max(s.index(c) for s in input_list)
                    for c in valid_next_chars
                }
                next_char = min(
                    valid_next_chars,
                    key=(lambda c: max_ind_dict[c]),
                )
                prefix_dict = {s: s.index(next_char) for s in input_list}
            else:
                prefix_dict = {s: len(s) for s in input_list}

            prefix_list = [s[:n] for s, n in prefix_dict.items()]
            input_list  = [s[n:] for s, n in prefix_dict.items()]
            output_str += str(sorted(set(prefix_list)))

        input_list = [s for s in input_list if len(s) > 0]

    if clean:
        output_str = clean_string(output_str).replace("_", "")

    return output_str

def get_unique_prefixes(
    input_list: list[str],
    forbidden: (set[str] | None)=None,
    min_len: int=1,
) -> dict[str, str]:
    if len(input_list) == 0:
        return dict()
    if forbidden is None:
        forbidden = set()

    remaining = set(input_list)
    prefix_dict = dict()
    max_len = max(len(s) for s in remaining)

    for i in range(min_len, max_len):
        partial_dict = {s: s[:i] for s in remaining}
        partial_list = list(partial_dict.values())
        new_prefixes = {
            s: p
            for s, p in partial_dict.items()
            if ((partial_list.count(p) == 1) and (p not in forbidden))
        }
        prefix_dict.update(new_prefixes)
        remaining -= set(new_prefixes.keys())
        if len(remaining) == 0:
            break

    prefix_dict.update({s: s for s in remaining})
    return prefix_dict
