#!/usr/bin/env python3
"""Real-time copycat guard — triggered by pull_request_target (opened).

Fires the instant a PR is opened, fingerprints its diff against every earlier
open PR that touches the same file(s), and responds with a graduated policy:

  ≥80% containment  →  instant block + close (zero tolerance — unchanged)
  70–79%            →  copycat-warn label + warning comment (no close, no block)
  2 warning strikes →  block + close (just like ≥80%)

Self-resubmissions (same author iterating on their own earlier PR) are excluded.
Only copycat detection runs here — no GPU, no eval, no scoring.

Invoked by .github/workflows/copycat-guard.yml via:
  PR_NUM=<num> python3 eval/copycat_guard.py
"""
import json, os, subprocess, sys
from datetime import date
from pathlib import Path

REPO = os.environ.get("EVAL_REPO", "gittensor-ai-lab/sparkinfer")
ROOT = Path(__file__).resolve().parents[1]
COPYCAT_LOG = ROOT / ".github" / "copycats.json"
DENYLIST_FILE = ROOT / ".github" / "blocked-contributors.txt"
FLAG_FILE = ROOT / ".github" / "FLAGGED.md"
COPYCAT_CONTAINMENT = 0.80        # ≥80% → instant block + close
COPYCAT_WARN         = 0.70        # 70–79% → warning label + comment (first time), block on second strike
MAX_WARNINGS         = 2           # block an account on this many warnings across any PRs
FLAG_LABEL = "flagged:gaming"


def gh(args):
    return subprocess.run(["gh"] + args, capture_output=True, text=True)


def pr_fingerprint(repo, num):
    """(changed files, normalized non-empty added lines) from the PR's unified diff."""
    diff = gh(["pr", "diff", str(num), "-R", repo]).stdout or ""
    files, added = set(), set()
    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            p = line[4:].strip()
            if p.startswith(("a/", "b/")): p = p[2:]
            if p and p != "/dev/null": files.add(p)
        elif line.startswith("+") and not line.startswith("+++"):
            s = line[1:].strip()
            if s and not s.startswith(("//", "#", "/*", "*")): added.add(s)
    return files, added


def containment(copy_added, orig_added):
    if not copy_added: return 0.0
    return len(copy_added & orig_added) / len(copy_added)


def load_denylist():
    try:
        out = set()
        for line in open(DENYLIST_FILE):
            s = line.split("#", 1)[0].strip().lower()
            if s: out.add(s)
        return out
    except Exception: return set()


def load_copycat_log():
    try: return json.load(open(COPYCAT_LOG))
    except Exception: return []


def save_copycat_log(log):
    COPYCAT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(COPYCAT_LOG, "w") as f: json.dump(log, f, indent=2)


def block_account(login, reason):
    cur = load_denylist()
    if login.lower() not in cur:
        with open(DENYLIST_FILE, "a") as f: f.write(f"\n{login}\n")
    with open(FLAG_FILE, "a") as f:
        f.write(f"\n## {date.today().isoformat()} — `{login}` (auto-blocked)\n\n{reason}\n")


def flag_copycat(repo, num, original, author):
    """≥80% containment — instant block + close (zero tolerance)."""
    subprocess.run(["gh", "pr", "edit", str(num), "-R", repo, "--add-label", "copycat"],
                   capture_output=True)
    body = (f"<!-- sparkinfer-copycat -->\n## 🐈 Flagged: copycat (real-time guard)\n\n"
            f"This PR re-submits substantially the same diff as the earlier #{original} by "
            f"a different author. Duplicating another contributor's work is treated as gaming "
            f"the SN74 emission mechanism. The account has been **blocked** and this PR "
            f"**closed** — zero tolerance, no warning.\n\n"
            f"See [`.github/COPYCATS.md`](../blob/main/.github/COPYCATS.md).")
    subprocess.run(["gh", "pr", "comment", str(num), "-R", repo, "--body", body], capture_output=True)


def warn_copycat(repo, num, original, author, strike_count, containment_pct):
    """70–79% containment — warning label + comment, no close, no block. Block on 2nd strike."""
    subprocess.run(["gh", "pr", "edit", str(num), "-R", repo, "--add-label", "copycat-warn"],
                   capture_output=True)
    will_block = (strike_count >= MAX_WARNINGS)
    action_line = ("\n\nThis is the **second** copycat-like submission — the account is now "
                   "**blocked** and the PR closed." if will_block
                   else f"\n\nThis is a **warning** (strike {strike_count}/{MAX_WARNINGS}). "
                   "A second copycat-like submission will result in an automatic block.\n\n"
                   "If this is a legitimate independent implementation, comment on this PR and "
                   "a maintainer will review.")
    body = (f"<!-- sparkinfer-copycat-warn -->\n## 🐈 Copycat warning (real-time guard)\n\n"
            f"This PR is **{containment_pct:.0f}% contained** in the earlier #{original} "
            f"by a different author, in the 70–79% warning range.{action_line}")
    subprocess.run(["gh", "pr", "comment", str(num), "-R", repo, "--body", body], capture_output=True)
    if will_block:
        close_blocked_pr(repo, num, {author})
    return will_block


def close_blocked_pr(repo, num, hits):
    subprocess.run(["gh", "pr", "edit", str(num), "-R", repo, "--add-label", FLAG_LABEL],
                   capture_output=True)
    who = ", ".join(f"`{h}`" for h in sorted(hits))
    body = ("<!-- sparkinfer-flagged -->\n"
            "## 🚩 Flagged: eval-gaming\n\n"
            f"This PR involves an account blocked for gaming the SN74 emission mechanism "
            f"(sybil / coordinated duplicate farming): {who}.\n\n"
            "Per the project's no-gaming policy these accounts are blocked: the PR is **not "
            "evaluated, scored, or merged**. See [`.github/FLAGGED.md`]"
            "(../blob/main/.github/FLAGGED.md) for the evidence and record.")
    subprocess.run(["gh", "pr", "comment", str(num), "-R", repo, "--body", body], capture_output=True)
    return subprocess.run(["gh", "pr", "close", str(num), "-R", repo]).returncode == 0


def pr_author_login(repo, num):
    info = json.loads(gh(["pr", "view", str(num), "-R", repo,
                          "--json", "author"]).stdout or "{}")
    return (info.get("author") or {}).get("login", "")


# ---- main (triggered by pull_request_target) ----
def main():
    pr_num = int(os.environ.get("PR_NUM") or 0)
    if not pr_num:
        print("PR_NUM not set — nothing to guard"); return

    author = pr_author_login(REPO, pr_num)
    print(f"copycat-guard: PR #{pr_num} by {author} — scanning for copycat ...")

    # 1) Already-blocked contributor? Skip — the scheduled bot handles it, double-blocking is noise.
    denylist = load_denylist()
    if author.lower() in denylist:
        print(f"  author {author} already in denylist — skip")
        return

    # 2) Fingerprint the new PR
    files, added = pr_fingerprint(REPO, pr_num)
    if not added:
        print(f"  no added lines to scan — not a copycat"); return

    # 3) Fetch all open PR numbers with earlier numbers (different author, not blocked, not copycat)
    open_prs = json.loads(gh(["pr", "list", "-R", REPO, "--state", "open",
                               "--json", "number,author,isDraft", "--limit", "100"]).stdout or "[]")
    log = load_copycat_log()
    blocked_prs = {e["pr"] for e in log}
    earlier_nums = sorted(p["number"] for p in open_prs if p["number"] < pr_num and not p["isDraft"])
    print(f"  {len(earlier_nums)} earlier open non-draft PRs to check")

    # 4) For each earlier PR touching shared files, fingerprint it. If >=70% containment
    #    -> graduated response: >=80% = instant block; 70-79% = warning (block on 2nd strike).
    original = None; orig_author = None; best_containment = 0.0
    for e_num in earlier_nums:
        e_author = next((p["author"]["login"] for p in open_prs if p["number"] == e_num), "")
        if not e_author or e_author == author: continue
        if e_author.lower() in denylist: continue
        if e_num in blocked_prs: continue
        ef, ea = pr_fingerprint(REPO, e_num)
        if not (files & ef): continue
        c = containment(added, ea)
        if c > best_containment:
            original = e_num; orig_author = e_author; best_containment = c
        if c >= COPYCAT_CONTAINMENT:
            break

    if original is None or best_containment < COPYCAT_WARN:
        print(f"  no copycat detected — clean"); return

    # 5) Graduated response: ≥80% immediate block; 70-79% warning (block on 2nd strike)
    is_block = (best_containment >= COPYCAT_CONTAINMENT)
    warn_strikes = sum(1 for e in log if e.get("author") == author and not e.get("blocked", True))
    strike_count = warn_strikes + 1

    if is_block:
        print(f"  COPYCAT (≥80%): #{pr_num} is {best_containment:.1%} contained in #{original} by {orig_author}")
        flag_copycat(REPO, pr_num, original, author)
        log.append({"pr": pr_num, "author": author, "original": original,
                    "date": date.today().isoformat(), "blocked": True})
        save_copycat_log(log)
        block_account(author, f"Auto-blocked: #{pr_num} is a copycat of #{original} "
                              f"(containment {best_containment:.0%}). "
                              f"Opened by {author}, copying {orig_author}'s unmerged work.")
        closed = close_blocked_pr(REPO, pr_num, {author})
        print(f"  copycat #{pr_num} flagged + blocked + closed={closed}")
    else:
        print(f"  COPYCAT WARNING (70-79%): #{pr_num} is {best_containment:.1%} contained in #{original} by {orig_author} (strike {strike_count}/{MAX_WARNINGS})")
        will_block = warn_copycat(REPO, pr_num, original, author, strike_count, best_containment)
        log.append({"pr": pr_num, "author": author, "original": original,
                    "date": date.today().isoformat(), "blocked": False,
                    "penalty_days": 0, "strike": strike_count,
                    "containment": round(best_containment, 3)})
        save_copycat_log(log)
        if will_block:
            block_account(author, f"Auto-blocked after {strike_count} copycat warnings "
                                  f"(latest: #{pr_num}, {best_containment:.0%} contained in #{original} "
                                  f"by {orig_author}). Two-strike rule.")
            # close after denylisting
            close_blocked_pr(REPO, pr_num, {author})

    # Push the updated github-policy files so the bot's run stays in sync
    subprocess.run(["git", "-C", str(ROOT), "add",
                    ".github/copycats.json", ".github/blocked-contributors.txt", ".github/FLAGGED.md"],
                   capture_output=True)
    if subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode != 0:
        subprocess.run(["git", "-C", str(ROOT), "commit", "-q",
                        "-m", f"copycat-guard: #{pr_num} flagged by {author}"],
                       capture_output=True)
        subprocess.run(["git", "-C", str(ROOT), "pull", "-q", "--rebase", "origin", "main"],
                       capture_output=True)
        subprocess.run(["git", "-C", str(ROOT), "push", "-q", "origin", "main"], capture_output=True)
        print("  policy files pushed")


if __name__ == "__main__":
    main()
