# SPDX-FileCopyrightText: 2026 Mike Volokhov
# SPDX-License-Identifier: LGPL-2.1-or-later

"""What the guards see when they ask which files are this project's.

Every test that polices the tree draws its list from :func:`repo.sources`, and
each is parametrized over it - so a list that had quietly stopped reaching part
of the tree is fewer cases rather than a failure, and reads from the outside
like a rule that holds everywhere. So the list itself is asserted here.

The case it exists for is the untracked one: a file is untracked for as long as
nobody has staged it, and the same word covers a download, an environment and a
build's output, which is why the corners below are as much of this file as the
rule is.

Each case is built as its own repository rather than looked for in this one: a
tree holding one file of each kind can be stated, and this one holds whatever
its author happens to have open. That is why :func:`cut_off` is here, and why
:func:`git` will not run without it.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

from tests import repo

#: Who the commits below are by, supplied per command because the environment
#: they run in has no config to read an identity from. Where git can guess one
#: off the system it does and this is redundant; where it cannot, it declines to
#: commit - so this is what keeps the file from depending on which.
AUTHOR = ("-c", "user.name=Test", "-c", "user.email=test@example.invalid")

#: Where every path this file points git at leads. Named so that :func:`git` can
#: recognise an environment :func:`cut_off` has been through, and refuse one it
#: has not.
NOWHERE = "nothing-of-anybody-elses"


def cut_off(monkeypatch, tmp_path: pathlib.Path) -> None:
    """Cut git off from everything outside this file, both ways.

    Outward first, because what this file does is ``init``, ``commit`` and
    ``merge``, and it may run inside a pre-commit hook. Git hands a hook an
    environment that redirects the next git run inside it, and which variables
    those are is not the guessable set: under git 2.50.1 an ordinary pre-commit
    hook gets ``GIT_INDEX_FILE`` as a relative path and ``GIT_CONFIG_PARAMETERS``
    carrying the outer ``-c`` flags, while ``GIT_DIR`` appears only in a linked
    worktree. So the whole namespace goes rather than the names one would think
    of.

    Inward so that a machine cannot decide these results. A name in the
    developer's own excludes file would otherwise leave a green run that proved
    nothing, and that door has more than one leaf: with no config naming one,
    git reads ``$XDG_CONFIG_HOME/git/ignore``, and ``$HOME/.config/git/ignore``
    only where that variable is unset - so pointing the first at nothing closes
    both.

    ``GIT_CEILING_DIRECTORIES`` is the one that has to be set: git otherwise
    walks up out of ``tmp_path`` looking for a repository, and finds one
    wherever ``TMPDIR`` or ``--basetemp`` points inside a checkout.
    """
    for name in [found for found in os.environ if found.startswith("GIT_")]:
        monkeypatch.delenv(name)
    nowhere = tmp_path / NOWHERE
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(nowhere))
    # What this one holds off is an `/etc/gitconfig` on the machine, which is on
    # disk rather than in the environment - so a test can show which file git is
    # standing at, and not what would follow from its standing at the other one.
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(nowhere))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(nowhere))


@pytest.fixture(autouse=True)
def hermetic(tmp_path, monkeypatch) -> None:
    cut_off(monkeypatch, tmp_path)


def git(*arguments: str, at: pathlib.Path) -> None:
    """Run git, having first made sure it cannot reach past this file.

    The check is here rather than left to the fixture, because what it stands
    against is the fixture ceasing to be automatic - which costs commits written
    into the repository this file is being run in, while the tests around them
    go on reporting that they passed.
    """
    assert os.environ.get("GIT_CONFIG_GLOBAL", "").endswith(NOWHERE), (
        "this git has not been cut off from the environment - see cut_off"
    )
    subprocess.run(["git", "-C", str(at), *AUTHOR, *arguments], check=True, capture_output=True)


def repository(at: pathlib.Path) -> pathlib.Path:
    at.mkdir(parents=True, exist_ok=True)
    git("init", "-q", str(at), at=at.parent)
    return at


@pytest.fixture
def tree(tmp_path) -> pathlib.Path:
    """A repository holding one file of every kind the rule distinguishes."""
    root = repository(tmp_path / "tree")
    (root / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (root / "_vendor").mkdir()

    for name in ("staged.py", "written.py", "ignored.py", "notes.md", "_vendor/theirs.py"):
        (root / name).write_text("x = 1\n", encoding="utf-8")

    git("add", "staged.py", "notes.md", at=root)
    return root


def listed(tree: pathlib.Path, *suffixes: str) -> list[str]:
    return [str(path.relative_to(tree)) for path in repo.sources(*suffixes, root=tree)]


def test_a_file_that_has_been_staged_is_this_projects(tree):
    assert "staged.py" in listed(tree, ".py")


def test_and_so_is_one_that_has_only_been_written(tree):
    """The case the rule exists for: read the index alone and the newest file in
    the tree is the one no guard has ever run over."""
    assert "written.py" in listed(tree, ".py")


def test_but_not_one_git_was_told_to_ignore(tree):
    """Which is what keeps a tool's leavings out with no list here naming them:
    an environment, a build's output and the caches are already named to git."""
    assert "ignored.py" not in listed(tree, ".py")


def test_nor_somebody_elses_text_vendored_into_the_tree(tree):
    assert str(pathlib.Path("_vendor", "theirs.py")) not in listed(tree, ".py")


def test_but_a_vendored_name_above_the_root_is_somebody_elses_business(tmp_path):
    """``_vendor`` names a place inside the tree that this project put somebody
    else's code in, so it is read from the root down. A directory of that name
    standing over the whole checkout is not this project's statement about
    anything."""
    root = repository(tmp_path / "_vendor" / "tree")
    (root / "ours.py").write_text("x = 1\n", encoding="utf-8")
    assert listed(root, ".py") == ["ours.py"]


def test_a_file_of_another_suffix_is_left_to_whoever_asks_for_that_one(tree):
    assert "notes.md" not in listed(tree, ".py")
    assert "notes.md" in listed(tree, ".md")


def test_a_repository_cloned_into_the_tree_is_not_descended_into(tree):
    """git lists an untracked repository as the one directory it stands in and
    stops there, so somebody else's clone arrives as a directory rather than as
    its text. Named to look like one of ours, since a directory can carry any
    name and the rule would otherwise be leaning on the suffix."""
    inner = repository(tree / "cloned.py")
    (inner / "theirs.py").write_text("x = 1\n", encoding="utf-8")
    assert not [name for name in listed(tree, ".py") if name.startswith("cloned")]


def test_but_a_file_of_ours_with_a_directory_over_it_still_does(tree):
    """A half-done rename, or a checkout stopped across a commit that turned a
    file into a directory. The entry has no trailing slash, so it is one of
    ours, and dropping it for being a directory in the working tree would take a
    tracked file out of every guard with nothing said."""
    git("add", "written.py", at=tree)
    (tree / "written.py").unlink()
    (tree / "written.py").mkdir()
    (tree / "written.py" / "inside.txt").write_text("x\n", encoding="utf-8")
    assert "written.py" in listed(tree, ".py")


def test_a_path_the_working_tree_does_not_have_comes_back_all_the_same(tree):
    """A rename half done, an interrupted rebase, a link whose target has gone.
    The guard reading it fails on it and says which path it was; dropping it
    here would leave that guard checking a smaller tree and saying nothing."""
    git("add", "written.py", at=tree)
    (tree / "written.py").unlink()
    assert "written.py" in listed(tree, ".py")


def test_a_file_in_conflict_is_returned_once(tree):
    """An unresolved merge holds three entries for one path - the base and the
    two sides - and a caller parametrized over this would collect three cases
    with one id between them."""
    git("commit", "-qm", "base", at=tree)
    git("checkout", "-qb", "other", at=tree)
    (tree / "staged.py").write_text("x = 2\n", encoding="utf-8")
    git("commit", "-qam", "theirs", at=tree)
    git("checkout", "-q", "-", at=tree)
    (tree / "staged.py").write_text("x = 3\n", encoding="utf-8")
    git("commit", "-qam", "ours", at=tree)
    subprocess.run(
        ["git", "-C", str(tree), *AUTHOR, "merge", "other"], check=False, capture_output=True
    )

    unmerged = repo.run_git("ls-files", "-u", at=tree)
    assert unmerged.count("staged.py") == 3, "the merge left nothing in conflict to count"
    assert listed(tree, ".py").count("staged.py") == 1


def test_the_list_comes_back_in_an_order_that_does_not_move(tmp_path):
    """Otherwise the ids a parametrized guard collects under shuffle between
    runs, and a failure cannot be pointed at.

    The names are enough of them, and written in enough of a jumble, that the
    order a set happens to iterate in will not pass for sorted. A pair of them
    would come out in order often enough for a broken implementation to go
    through.
    """
    root = repository(tmp_path / "jumbled")
    names = ["quince.py", "apple.py", "medlar.py", "cherry.py", "sloe.py", "damson.py", "fig.py"]
    for name in names:
        (root / name).write_text("x = 1\n", encoding="utf-8")
    assert listed(root, ".py") == sorted(names)


def test_a_directory_that_is_no_repository_refuses_rather_than_answering_nothing(tmp_path):
    """An empty list is what every caller here would read as a rule holding
    everywhere, so the one thing this may not do is return one quietly."""
    outside = tmp_path / "not-a-repository"
    outside.mkdir()
    with pytest.raises(RuntimeError, match="git said"):
        repo.sources(".py", root=outside)


def test_and_so_does_having_no_git_to_ask(tree, monkeypatch):
    """The same refusal by the other route, which arrives as an ``OSError`` from
    the process rather than as a status from git. The two say which they are:
    one refusal covering both would let either be reported as the other."""
    monkeypatch.setenv("PATH", str(tree / "nothing-here"))
    with pytest.raises(RuntimeError, match="no git to ask"):
        repo.sources(".py", root=tree)


def test_a_repository_further_up_is_not_borrowed(tmp_path, monkeypatch):
    """git walks upward looking for one, and the directory a test works in is
    inside a checkout wherever somebody has pointed ``TMPDIR`` or ``--basetemp``
    at one. Built here as that shape - a checkout, and the temporary directory
    under it - because without a ceiling the case above finds that repository,
    answers out of it, and stops testing anything."""
    outer = repository(tmp_path / "checkout-somebody-pointed-at")
    temporary = outer / "temporary"
    temporary.mkdir()
    cut_off(monkeypatch, temporary)

    plain = temporary / "not-a-repository-either"
    plain.mkdir()
    with pytest.raises(RuntimeError, match="git said"):
        repo.sources(".py", root=plain)


def test_the_repositories_here_cannot_reach_one_outside_them(tmp_path, monkeypatch):
    """What ``cut_off`` is for, driven the way a hook would drive it.

    The victim is pointed at by the whole set of variables git hands a hook, and
    then this file does its ordinary work beside it. Without the clearing, the
    ``init``, the commits and the branch land in the victim instead.
    """
    victim = repository(tmp_path / "victim")
    (victim / "work.py").write_text("x = 1\n", encoding="utf-8")
    git("add", "work.py", at=victim)
    git("commit", "-qm", "the only real commit", at=victim)
    was = repo.run_git("log", "--oneline", at=victim), repo.run_git("branch", at=victim)

    monkeypatch.setenv("GIT_DIR", str(victim / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(victim))
    monkeypatch.setenv("GIT_INDEX_FILE", str(victim / ".git" / "index"))
    cut_off(monkeypatch, tmp_path)

    beside = repository(tmp_path / "beside")
    (beside / "theirs.py").write_text("x = 1\n", encoding="utf-8")
    git("add", "theirs.py", at=beside)
    git("commit", "-qm", "not the victim's", at=beside)
    git("checkout", "-qb", "a-branch-of-its-own", at=beside)

    assert (repo.run_git("log", "--oneline", at=victim), repo.run_git("branch", at=victim)) == was


def test_nor_an_excludes_file_the_machine_happens_to_carry(tmp_path, monkeypatch):
    """Where git reads one from when no config names one. A developer with
    ``written.py`` in theirs would otherwise watch the case above pass while the
    list it asserts had lost the entry it is about."""
    carried = tmp_path / "somebody-elses-config" / "git"
    carried.mkdir(parents=True)
    (carried / "ignore").write_text("written.py\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(carried.parent))

    cut_off(monkeypatch, tmp_path)
    root = repository(tmp_path / "excluded")
    (root / "written.py").write_text("x = 1\n", encoding="utf-8")
    assert listed(root, ".py") == ["written.py"]


def test_and_the_system_config_is_one_this_file_named(tmp_path):
    """The door no test can open, seen from the outside. What follows from a
    machine carrying an ``/etc/gitconfig`` cannot be shown without writing one,
    but which file git is standing at can be, because it names that file when it
    cannot read it."""
    said = subprocess.run(
        ["git", "-C", str(tmp_path), "config", "--system", "--list"],
        capture_output=True,
        text=True,
    )
    assert NOWHERE in said.stderr, f"git read its system config from somewhere else: {said.stderr}"


def test_and_a_git_that_has_not_been_cut_off_does_not_run_at_all(tmp_path, monkeypatch):
    """What makes the fixture's automatic-ness loud rather than assumed. The
    fixture and this check hold each other up: either alone still fails
    something, and both gone leaves commits and a branch in the repository the
    suite was run in, with the tests around them green."""
    monkeypatch.delenv("GIT_CONFIG_GLOBAL")
    with pytest.raises(AssertionError, match="cut off from the environment"):
        git("init", "-q", str(tmp_path / "never-built"), at=tmp_path)


def test_on_this_tree_the_list_reaches_the_workbench():
    """Named rather than counted, because the number moves with every file
    added. Without it, a list that had stopped reaching anything - or a default
    root pointing somewhere else - would leave every guard drawing from it
    green."""
    assert repo.ROOT / "Microwave" / "units.py" in repo.sources(".py")
