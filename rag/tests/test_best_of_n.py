import threading

from rag.best_of_n import best_of_n, vote


def test_vote_groups_the_same_number_written_differently():
    v = vote(["-2%", "-2.04%", "-2.0%", "$5.4 billion", None])
    assert (v["answer"], v["votes"], v["n"]) == ("-2%", 3, 5)        # -2.04 rounds to the coarser -2
    assert [g["votes"] for g in v["groups"]] == [3, 1]                 # no-answer runs never form a group


def test_vote_groups_across_scale_words():
    assert vote(["$260.2 billion", "260,174 million"])["votes"] == 2  # within 1%


def test_vote_keeps_different_numbers_apart_and_ties_go_to_the_first():
    v = vote(["-3%", "-2%", "-2.6%"])                                  # -2.6 rounds to -3, not -2
    assert (v["answer"], v["votes"]) == ("-3%", 2)
    assert vote(["-2%", "-3%"])["answer"] == "-2%"


def test_vote_reads_the_number_as_written():
    """The prompt asks for a decrease as a negative number; "decreased 2%" reads as +2."""
    assert vote(["Net sales decreased 2%", "-2%"])["votes"] == 1


def test_vote_with_no_numbers():
    v = vote(["not found", None, "not found"])
    assert (v["answer"], v["value"], v["votes"], v["groups"]) == (None, None, 0, [])


class FakeModel:
    """Answers by seed; records every call's sampling settings. A barrier proves the runs are
    sent at once (so vLLM can batch them): 5 runs one after another would time out."""
    def __init__(self, answers, n=5):
        self.answers, self.calls, self.barrier = answers, [], threading.Barrier(n, timeout=5)

    def __call__(self, messages, temperature, top_p, seed):
        self.calls.append((seed, temperature, top_p))
        self.barrier.wait()
        if self.answers[seed] == "crash":
            raise RuntimeError("server answered 500")
        return f"Answer: {self.answers[seed]}"


def search(query, k):
    return [{"company": "Apple Inc.", "year": 2019, "section": "7", "chunk": 1, "text": "Net sales decreased 2%"}]


def test_best_of_n_runs_all_at_once_with_seeds_and_votes():
    model = FakeModel({0: "-2%", 1: "-2.04%", 2: "-3%", 3: "-2.0%", 4: "not found"})
    r = best_of_n("What was Apple's revenue growth in 2019?", model, search)
    assert sorted(model.calls) == [(s, 0.7, 0.95) for s in range(5)]
    assert [run["seed"] for run in r["runs"]] == [0, 1, 2, 3, 4]
    assert [run["answer"] for run in r["runs"]] == ["-2%", "-2.04%", "-3%", "-2.0%", "not found"]
    assert (r["vote"]["answer"], r["vote"]["votes"], r["vote"]["n"]) == ("-2%", 3, 5)


def test_a_crashed_run_counts_as_no_answer():
    r = best_of_n("q", FakeModel({0: "-2%", 1: "crash", 2: "-2%", 3: "-2%", 4: "-2%"}), search)
    assert r["runs"][1]["stopped"] == "error" and "500" in r["runs"][1]["error"]
    assert (r["vote"]["votes"], r["vote"]["n"]) == (4, 5)


def test_same_answer_uses_the_votes_rule():
    from rag.best_of_n import same_answer
    assert same_answer("-2%", "-2.0%") and same_answer("$260.2 billion", "260,174 million")
    assert not same_answer("-2%", "-3%")
    assert not same_answer(None, "-2%") and not same_answer("not found", "not found")


def test_a_year_after_the_answer_is_not_the_answer():
    """Review finding 6: the last number was read, so "$641 million in fiscal 2019" read as 2019."""
    from rag.best_of_n import same_answer
    assert same_answer("$641 million in fiscal 2019", "$641 million")
    assert same_answer("$557 million for fiscal year 2018", "$557 million")
    assert same_answer("-2% in 2019", "-2%") and same_answer("21% in FY2019", "21%")
    assert vote(["$641 million in fiscal 2019", "$641 million", "$641 million during 2019"])["votes"] == 3
    assert same_answer("2019", "2019") and not same_answer("2019", "2018")     # a year alone stays a year


def test_a_percent_and_its_fraction_are_the_same_answer():
    """Review finding 7: -0.02 and -2% never matched."""
    from rag.best_of_n import same_answer
    assert same_answer("-0.02", "-2%") and same_answer("0.72", "72%") and same_answer("0.08", "8 percent")
    assert same_answer("-0.0204", "-2%")                    # rounding still forgiven
    assert same_answer("-2.04%", "-2%") and not same_answer("-2%", "-3%")
    assert not same_answer("2", "2%")                       # a bare 2 is not read as 2%


def test_zero_is_not_a_rounded_answer():
    """Found re-checking v3's real replies: its "0" for Netflix (worked out from "subtract 2,887 and
    2,887") was marked right against 28%: 0 has no decimals, and 0.28 rounded to none is 0. Rounding
    to 0 says nothing, so 0 must match within 1% like any number."""
    from rag.best_of_n import same_answer
    assert not same_answer("0", "28%") and not same_answer("0.0", "0.28") and not same_answer("0", "-0.4")
    assert vote(["0", "0.28", "0", "0"])["votes"] == 3
    assert same_answer("0", "0.0") and same_answer("0%", "0")
    assert same_answer("-2.04%", "-2%") and same_answer("3", "2.6")       # the rounding rule otherwise stands
