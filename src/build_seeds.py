"""Build personas/seeds.json -- the domain-conditioned seed library (Phase 6.1).

Gap G2: TVSum and SumMe span very different domains, so a single global persona
distribution is wrong. Phase 5 demonstrated the failure empirically -- a pastry-chef
persona rating a tyre-changing tutorial returned a CONSTANT all-zero score vector, which
carries zero learning signal and makes rank correlation undefined.

So personas are generated per domain, and Phase 7 only pairs a persona with videos from
its own domain.

Domains:
  TVSum -- its own 10 native categories (5 videos each), read from
           ydata-tvsum50-info.tsv, joined to h5 keys via data/video_map.json
           (mp4 stem == TVSum video_id).
  SumMe -- has no category field, so its 25 videos are assigned explicitly below from
           their official titles. Hand-assigned but version-controlled and auditable.

The per-domain lists are the axes a persona varies along; gen_personas.py samples from
them and asks the teacher to expand a sampled combination into a coherent viewer.

Usage:  python src/build_seeds.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TVSUM_INFO = Path("/var/tmp/akn57/data/raw/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv")

# --- TVSum's 10 native categories -------------------------------------------------
TVSUM_DOMAINS = {
    "VT": dict(label="vehicle tyre repair (instructional)",
               viewing_goal=["follow the exact repair steps", "decide if I can do it myself",
                             "check which tools are needed", "quick overview before starting"],
               interest_focus=["hands working on the wheel", "tools and equipment",
                               "the finished result", "safety warnings", "the presenter explaining"]),
    "VU": dict(label="vehicles stuck, unusual or in trouble",
               viewing_goal=["see the dramatic moment", "understand what went wrong",
                             "learn the recovery technique", "judge the vehicle's capability"],
               interest_focus=["the moment of failure", "recovery attempts", "the terrain",
                               "the vehicle itself", "people reacting"]),
    "BK": dict(label="beekeeping and wasp removal",
               viewing_goal=["learn hive management", "see the insects up close",
                             "assess the danger", "follow the removal procedure"],
               interest_focus=["close-ups of bees", "hive structure", "protective equipment",
                               "the beekeeper's technique", "honey and combs"]),
    "BT": dict(label="bike and motorcycle tricks (how-to and stunt shows)",
               viewing_goal=["learn to perform the trick", "watch the best stunts",
                             "check the bike setup", "see crashes and fails"],
               interest_focus=["the trick being landed", "body position and technique",
                               "the bike and its parts", "crowd reaction", "the instructor talking"]),
    "DS": dict(label="dog shows and pet stories",
               viewing_goal=["see the dogs", "follow the competition result",
                             "understand judging criteria", "follow one animal's story"],
               interest_focus=["individual dogs", "handlers and judges", "the crowd and venue",
                               "close-ups of animals", "the award moment"]),
    "GA": dict(label="grooming an animal",
               viewing_goal=["learn the grooming technique", "decide on a service for my pet",
                             "see the before/after", "check the tools used"],
               interest_focus=["the groomer's hands", "the animal's reaction", "tools and clippers",
                               "the finished look", "the salon environment"]),
    "MS": dict(label="making a sandwich (food preparation)",
               viewing_goal=["replicate the recipe exactly", "decide whether to order it",
                             "see the ingredients", "quick overview of the dish"],
               interest_focus=["ingredients being added", "hands preparing food", "the finished dish",
                               "the cook talking", "the restaurant or kitchen"]),
    "PK": dict(label="parkour and free running",
               viewing_goal=["study the movement technique", "watch the most impressive jumps",
                             "see the locations used", "follow one athlete"],
               interest_focus=["the jump or vault itself", "landings and falls", "the urban setting",
                               "athletes talking", "group interaction"]),
    "PR": dict(label="parades and street processions",
               viewing_goal=["see the whole spectacle", "spot a specific group or float",
                             "sense the atmosphere", "catch the main highlight"],
               interest_focus=["floats and costumes", "performers and dancers", "the crowd",
                               "the street and surroundings", "close-ups of participants"]),
    "FM": dict(label="flash mobs and staged public performances",
               viewing_goal=["watch the performance", "see the crowd's reaction",
                             "understand the message or cause", "catch the surprise moment"],
               interest_focus=["the dancers", "the moment it starts", "bystanders reacting",
                               "banners and signage", "the full group formation"]),
}

# --- SumMe: assigned from official titles ----------------------------------------
SUMME_DOMAINS = {
    "action_sports": dict(label="extreme and action sports",
                          videos=["Base jumping", "Bearpark_climbing", "Bike Polo", "Jumps",
                                  "Paintball", "Playing_on_water_slide", "Scuba",
                                  "Valparaiso_Downhill", "paluma_jump", "playing_ball"],
                          viewing_goal=["see the peak action moment", "study the technique",
                                        "sense the whole experience", "watch for crashes or fails"],
                          interest_focus=["the athlete in motion", "the landing or impact",
                                          "the scenery", "equipment", "people reacting"]),
    "aviation": dict(label="aviation and flight",
                     videos=["Air_Force_One", "Cockpit_Landing", "St Maarten Landing",
                             "Uncut_Evening_Flight"],
                     viewing_goal=["watch the landing or takeoff", "see the cockpit procedure",
                                   "spot the aircraft type", "enjoy the views"],
                     interest_focus=["the aircraft", "the cockpit and instruments", "the runway",
                                     "the view from the window", "ground crew and spectators"]),
    "vehicles": dict(label="vehicles and machinery in unusual situations",
                     videos=["Bus_in_Rock_Tunnel", "Car_railcrossing",
                             "Excavators river crossing", "car_over_camera"],
                     viewing_goal=["see the risky manoeuvre", "understand what happened",
                                   "judge the machine's capability", "catch the dramatic moment"],
                     interest_focus=["the vehicle", "the obstacle or terrain", "the critical moment",
                                     "bystanders", "the aftermath"]),
    "landmarks_travel": dict(label="landmarks and travel",
                             videos=["Eiffel Tower", "Notre_Dame", "Statue of Liberty"],
                             viewing_goal=["see the landmark itself", "plan my own visit",
                                           "sense the atmosphere", "see architectural detail"],
                             interest_focus=["the monument", "architectural close-ups", "the crowd",
                                             "the surrounding city", "people in the frame"]),
    "everyday_life": dict(label="everyday life and family moments",
                          videos=["Kids_playing_in_leaves", "Cooking", "Fire Domino"],
                          viewing_goal=["relive the moment", "see the people I know",
                                        "follow what is being made", "catch the funny bit"],
                          interest_focus=["faces and expressions", "the activity itself",
                                          "the result", "the setting", "spontaneous moments"]),
    "animals_nature": dict(label="animals and nature",
                           videos=["Saving dolphines"],
                           viewing_goal=["see the animals", "follow the rescue or event",
                                         "understand what is happening", "see the environment"],
                           interest_focus=["the animals", "people helping", "the water or landscape",
                                           "close-ups", "the outcome"]),
}

# --- attributes varied for every persona, in every domain -------------------------
GLOBAL_ATTRIBUTES = dict(
    age_band=["18-24", "25-34", "35-49", "50-64", "65+"],
    attention_budget=["15s", "30s", "1min", "2min"],
    expertise=["complete novice", "casual interest", "hobbyist", "domain professional"],
    watching_context=["deciding whether to watch the full video", "on a phone during a commute",
                      "researching before doing it themselves", "relaxing at home",
                      "looking for one specific moment"],
    # Sampled explicitly rather than left to the model. The first audit of the pool found
    # 84% she/her vs 9% he/him when the model was free to choose -- a skew that would sit
    # unexamined in the persona pool. Sampling it makes the distribution controlled and
    # auditable; it does not affect what the persona wants to SEE.
    gender_presentation=["a woman", "a man", "a non-binary person"],
)


def tvsum_video_domains():
    """h5 key -> TVSum category, joined via the mp4 stem (== TVSum video_id)."""
    cat_by_id = {}
    with open(TVSUM_INFO) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            cat_by_id[row["video_id"]] = row["category"]
    vmap = json.load(open(ROOT / "data/video_map.json"))["tvsum"]
    out = {}
    for k, d in vmap.items():
        stem = Path(d["path"]).stem
        if stem not in cat_by_id:
            raise KeyError(f"{k}: mp4 stem '{stem}' not found in TVSum info tsv")
        out[k] = cat_by_id[stem]
    return out


def summe_video_domains():
    """h5 key -> assigned domain, via video_name."""
    name_to_dom = {v: dom for dom, d in SUMME_DOMAINS.items() for v in d["videos"]}
    vmap = json.load(open(ROOT / "data/video_map.json"))["summe"]
    out, missing = {}, []
    for k, d in vmap.items():
        dom = name_to_dom.get(d["video_name"])
        if dom is None:
            missing.append(f"{k} ({d['video_name']})")
        else:
            out[k] = dom
    if missing:
        raise KeyError(f"unassigned SumMe videos: {missing}")
    return out


def main():
    tv, sm = tvsum_video_domains(), summe_video_domains()
    domains = {f"tvsum:{k}": dict(dataset="tvsum", **v) for k, v in TVSUM_DOMAINS.items()}
    for k, v in SUMME_DOMAINS.items():
        d = {kk: vv for kk, vv in v.items() if kk != "videos"}
        domains[f"summe:{k}"] = dict(dataset="summe", **d)

    seeds = dict(
        note=("Domain-conditioned persona seeds (gap G2). Personas are generated per domain "
              "and paired only with videos of that domain -- Phase 5 showed mismatched "
              "personas produce constant, useless score vectors."),
        global_attributes=GLOBAL_ATTRIBUTES,
        domains=domains,
        video_domain={"tvsum": {k: f"tvsum:{v}" for k, v in tv.items()},
                      "summe": {k: f"summe:{v}" for k, v in sm.items()}},
    )
    out = ROOT / "personas/seeds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(seeds, indent=2))

    print(f"wrote {out}")
    print(f"  domains: {len(domains)}  ({sum(1 for d in domains if d.startswith('tvsum'))} tvsum, "
          f"{sum(1 for d in domains if d.startswith('summe'))} summe)")
    from collections import Counter
    print("  tvsum videos per domain:", dict(Counter(tv.values())))
    print("  summe videos per domain:", dict(Counter(sm.values())))
    combos = 1
    for v in GLOBAL_ATTRIBUTES.values():
        combos *= len(v)
    print(f"  global attribute combinations: {combos} (x goal x focus per domain)")


if __name__ == "__main__":
    main()
