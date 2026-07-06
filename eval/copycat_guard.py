#!/usr/bin/env python3
"""Real-time copycat guard — triggered by pull_request_target (opened).

Fires the instant a PR is opened, fingerprints its diff against every earlier
open PR that touches the same file(s), and if ≥80% of the new PR's added lines
are contained in an earlier PR by a DIFFERENT author, the new PR is flagged as a
copycat, the author is blocked, and the PR is closed — all within seconds of
creation, before the scheduled eval bot ever sees it.

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
COPYCAT_CONTAINMENT = 0.80
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
    subprocess.run(["gh", "pr", "edit", str(num), "-R", repo, "--add-label", "copycat"],
                   capture_output=True)
    body = (f"<!-- sparkinfer-copycat -->\n## 🐈 Flagged: copycat (real-time guard)\n\n"
            f"This PR re-submits substantially the same diff as the earlier #{original} by "
            f"a different author. Duplicating another contributor's work is treated as gaming "
            f"the SN74 emission mechanism. The account has been **blocked** and this PR "
            f"**closed** — zero tolerance, no warning.\n\n"
            f"See [`.github/COPYCATS.md`](../blob/main/.github/COPYCATS.md).")
    subprocess.run(["gh", "pr", "comment", str(num), "-R", repo, "--body", body], capture_output=True)


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

    # 4) For each earlier PR touching shared files, fingerprint it. If ≥80% containment -> copycat.
    original = None
    for e_num in earlier_nums:
        e_author = next((p["author"]["login"] for p in open_prs if p["number"] == e_num), "")
        if not e_author or e_author == author: continue
        if e_author.lower() in denylist: continue
        if e_num in blocked_prs: continue
        ef, ea = pr_fingerprint(REPO, e_num)
        if not (files & ef): continue
        c = containment(added, ea)
        if c >= COPYCAT_CONTAINMENT:
            original = e_num
            orig_author = e_author
            print(f"  COPYCAT DETECTED: #{pr_num} is {c:.1%} contained in #{e_num} by {e_author}")
            break

    if original is None:
        print(f"  no copycat detected — clean"); return

    # 5) Full treatment: label + comment + block + close
    flag_copycat(REPO, pr_num, original, author)
    log.append({"pr": pr_num, "author": author, "original": original,
                "date": date.today().isoformat()})
    save_copycat_log(log)
    block_account(author, f"Auto-blocked: #{pr_num} is a copycat of #{original} "
                          f"(containment {containment(added, ea):.0%}). "
                          f"Opened by {author}, copying {orig_author}'s unmerged work.")
    closed = close_blocked_pr(REPO, pr_num, {author})
    print(f"  copycat #{pr_num} flagged + blocked + closed={closed}")

    # Push the updated github-policy files so the bot's run stays in sync
    subprocess.run(["git", "-C", str(ROOT), "add",
                    ".github/copycats.json", ".github/blocked-contributors.txt", ".github/FLAGGED.md"],
                   capture_output=True)
    if subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode != 0:
        subprocess.run(["git", "-C", str(ROOT), "commit", "-q",
                        "-m", f"copycat-guard: #{pr_num} flagged as copycat of #{original} by {author}"],
                       capture_output=True)
        subprocess.run(["git", "-C", str(ROOT), "pull", "-q", "--rebase", "origin", "main"],
                       capture_output=True)
        subprocess.run(["git", "-C", str(ROOT), "push", "-q", "origin", "main"], capture_output=True)
        print("  policy files pushed")


if __name__ == "__main__":
    main()
