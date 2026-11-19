from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np

from .config import load_config
from .db import log_ingest, open_db
from .ingest import fixtures as fx
from .ingest import footballdata as fd
from .names import (
    SOURCE_FOOTBALLDATA,
    load_discovered,
    load_overrides,
    record_discovered,
    save_discovered,
    seed_foundation,
    unresolved_aliases,
    write_suggestions,
)
from .models.elo import load_tuning as load_elo_tuning
from .eval.tuning import load_dc_tuning


def _season_codes(first_season: str, current: str) -> list[str]:
    codes = []
    y0 = int(first_season[:2])
    y1 = int(current[:2])
    for y in range(y0, y1 + 1):
        codes.append(f"{y:02d}{(y + 1) % 100:02d}")
    return codes


def cmd_ingest(args) -> int:
    cfg = load_config(args.config)
    cfg.ensure_dirs()
    conn = open_db(cfg.db_path)
    applied = load_overrides(conn, cfg.alias_overrides_path)
    if applied:
        print(f"applied {applied} alias override(s) from {cfg.alias_overrides_path}")

    leagues = list(cfg.leagues.keys())
    current = fd.current_season_code()
    seasons = _season_codes(cfg.first_season, current)
    delay = float(cfg.data.get("request_delay_s", 1.0))
    discovered = load_discovered(cfg.discovered_names_path)

    print(f"seasons {seasons[0]}..{seasons[-1]}  leagues {','.join(leagues)}  "
          f"current season {current}  discover={args.discover}")
    print("-" * 100)

    totals = {"rows": 0, "rejected": 0, "files": 0, "missing": 0}
    per_league_names: dict[str, set] = {lg: set() for lg in leagues}

    for season in seasons:
        for league in leagues:
            result = fd.download_season(cfg, season, league, force=args.force)
            if result.status in ("missing", "error"):
                totals["missing"] += 1
                log_ingest(conn, "footballdata", f"{season}_{league}", 0, 0,
                           f"{result.status}: {result.detail}")
                print(f"{league} {season}: {result.status} ({result.detail})")
                time.sleep(delay)
                continue
            frame, report = fd.parse_season_file(result.path, league, season)
            totals["rows"] += report.rows_kept
            totals["files"] += 1

            if args.discover:
                for name in set(frame["home_name"].tolist()) | set(frame["away_name"].tolist()):
                    record_discovered(discovered, SOURCE_FOOTBALLDATA, str(name), league, season)
                per_league_names[league].update(str(n) for n in set(frame["home_name"]) | set(frame["away_name"]))
            else:
                stats = fd.upsert_matches(conn, frame)
                totals["rejected"] += stats["rejected"]
                note = "; ".join(report.notes)
                log_ingest(conn, "footballdata", f"{season}_{league}", stats["rows"], stats["rejected"], note)
                unresolved_note = ""
                if stats["unresolved"]:
                    unresolved_note = f"  unresolved names: {stats['unresolved'][:5]}"
                print(f"{league} {season}: {stats['rows']} rows, inserted {stats['inserted']}, "
                      f"rejected {stats['rejected']}{unresolved_note}")
            time.sleep(delay)

    path, status = fx.download_fixtures(cfg, force=args.force)
    if path is not None:
        frame = fx.parse_fixtures(path, leagues)
        if args.discover:
            for _, row in frame.iterrows():
                for nm in (row["home_name"], row["away_name"]):
                    record_discovered(discovered, SOURCE_FOOTBALLDATA, str(nm),
                                       str(row["league"]), str(row["season"]))
        else:
            stats = fx.upsert_fixtures(conn, frame)
            log_ingest(conn, "footballdata", "fixtures", stats["rows"], stats["rejected"], status)
            print(f"fixtures: {stats['rows']} rows, rejected {stats['rejected']}")
    else:
        print(f"fixtures: {status}")

    save_discovered(cfg.discovered_names_path, discovered)

    print("-" * 100)
    if args.discover:
        print(f"discovered names for {len(discovered.get(SOURCE_FOOTBALLDATA, {}))} clubs across "
              f"{totals['files']} files, {totals['rows']} rows parsed")
        for lg in leagues:
            names = sorted(per_league_names.get(lg, []))
            if names:
                print(f"\n{lg} ({len(names)}): " + ", ".join(names))
        print(f"\nNow run: python -m prorrhesis names suggest")
    else:
        row = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        print(f"total rows ingested: {totals['rows']}, rejected {totals['rejected']}, "
              f"matches in db: {row}, files missing: {totals['missing']}")
    conn.close()
    return 0


def cmd_clubelo_fetch(args) -> int:
    """Fetch ClubElo snapshots. The service is optional: a failure disables the feature."""
    from .eval.walkforward import load_matches
    from .ingest import clubelo as ce

    cfg = load_config(args.config)
    conn = open_db(cfg.db_path)
    try:
        matches = load_matches(conn)
        if len(matches) == 0:
            print("no matches ingested yet - run 'python -m prorrhesis ingest' first")
            return 1
        print(f"fetching ClubElo snapshots covering {len(matches)} matches...")
        report = ce.fetch_for_matches(conn, cfg, matches, force=args.force)
        if report.failed and not report.reachable:
            note = "; ".join(report.errors) or "service unreachable"
            log_ingest(conn, "clubelo", "snapshot", report.rows, report.failed, note)
            print(f"ClubElo is unreachable ({note}).")
            print("The clubelo feature is disabled for this run; the app keeps working without it. "
                  "Re-run 'python -m prorrhesis clubelo fetch' later to retry.")
            return 0
        log_ingest(conn, "clubelo", "snapshot", report.rows, report.failed,
                   "; ".join(report.errors)[:500])
        print(f"ClubElo: {report.fetched} fetched, {report.cached} cached, {report.failed} failed, "
              f"{report.rows} ratings stored")
        if report.unresolved:
            print(f"  {len(report.unresolved)} club name(s) did not resolve, e.g. "
                  f"{report.unresolved[:5]} - run 'python -m prorrhesis names suggest'")
        return 0
    finally:
        conn.close()


def cmd_names_suggest(args) -> int:
    cfg = load_config(args.config)
    conn = open_db(cfg.db_path)
    applied = load_overrides(conn, cfg.alias_overrides_path)
    if applied:
        print(f"applied {applied} alias override(s)")

    discovered = load_discovered(cfg.discovered_names_path)
    if not discovered:
        print("no discovered names yet - run: python -m prorrhesis ingest --discover")
        return 1

    report = seed_foundation(conn, discovered)
    unresolved = unresolved_aliases(conn, discovered)
    suggestions = cfg.data_dir / "alias_suggestions.txt"
    write_suggestions(suggestions, conn, unresolved)

    print(f"teams: {conn.execute('SELECT COUNT(*) FROM teams').fetchone()[0]}")
    print(f"aliases: {conn.execute('SELECT COUNT(*) FROM team_aliases').fetchone()[0]}")
    print(f"new teams: {report.created_teams}, new aliases: {report.new_aliases}")
    if report.fuzzy_matches:
        print(f"fuzzy mappings applied (logged for review): {len(report.fuzzy_matches)}")
        for alias, source, canon, target, _ in report.fuzzy_matches[:20]:
            print(f"  {source}: '{alias}' -> {target} (slug {canon})")
    print(f"unresolved: {len(unresolved)}")
    if unresolved:
        for alias, source in unresolved[:20]:
            print(f"  {source}: {alias}")
        print(f"suggestions written to {suggestions}")
    conn.close()
    return 0 if not unresolved else 0


def cmd_backtest(args) -> int:
    from .eval.holdout import holdout_broken, read_lock, season_windows
    from .eval.walkforward import (
        arena_diagnostics,
        arena_table,
        compare_models,
        create_run,
        walk_forward,
    )
    from .models import build_models

    cfg = load_config(args.config)
    conn = open_db(cfg.db_path)
    win = season_windows(conn, cfg)

    if args.kind == "tuning":
        start, end = win.tuning_start, win.tuning_end
    else:
        start, end = win.headline_start, win.headline_end
        lock = read_lock(cfg)
        if not lock:
            print("no holdout.lock - run: python -m prorrhesis holdout declare")
            return 1
        if lock.get("config_hash") != cfg.config_hash:
            print("config changed since the holdout was declared - the headline run would be "
                  "'holdout_broken'. Re-declare a holdout to proceed.")
            return 1

    tuning = load_elo_tuning(cfg)
    if args.tune:
        from .eval.walkforward import load_matches
        from .models.elo import save_tuning, tune_elo

        burn_in = load_matches(conn)
        burn_in = burn_in[burn_in["date"] < str(cfg.eval.get("eval_start", "2017-08-01"))]
        print(f"tuning Elo on burn-in ({len(burn_in)} matches before {cfg.eval.get('eval_start')})...")
        tuning = tune_elo(cfg, burn_in)
        save_tuning(cfg, tuning)
        print(f"tuned: K={tuning['K']} rho={tuning['rho']} H={tuning['H']}")

    dc_tuning = load_dc_tuning(cfg)
    if args.tune_dc:
        from .eval.tuning import save_dc_tuning, tune_dc
        from .eval.walkforward import load_matches

        history = load_matches(conn)
        print(f"tuning Dixon-Coles on the tuning window ({start} .. {end})...")
        dc_tuning = tune_dc(cfg, history, start, end)
        save_dc_tuning(cfg, dc_tuning)
        print(f"tuned dc: {dc_tuning['half_life_days']:.0f} days, gamma={dc_tuning['gamma']}, "
              f"window={dc_tuning['window_seasons']} seasons -> logloss {dc_tuning['log_loss']:.4f}")

    model_names = args.models.split(",") if args.models else ["freq", "elo", "market_close"]
    models = build_models(cfg, model_names, tuning=tuning, dc_tuning=dc_tuning)

    run_id = f"{args.kind}_{cfg.config_hash[:8]}"
    create_run(conn, run_id, args.kind, cfg, notes=f"{start}..{end}")
    print(f"run {run_id}: {start} .. {end}  models={','.join(model_names)}")
    ensemble_members = None
    if args.ensemble:
        # the closing-odds benchmark may never feed the stacker (I5)
        ensemble_members = [n for n in model_names if n != "market_close"]

    result = walk_forward(conn, models, cfg, run_id, start, end, progress=True,
                          ensemble_members=ensemble_members)
    print(f"\n{result.blocks} blocks, {result.matches} matches, {result.seconds:.0f}s")
    if result.failed:
        print(f"model failures: {result.failed}")
    if result.ensemble:
        print("ensemble methods: " + ", ".join(f"{k}={v}" for k, v in sorted(result.ensemble.items())))

    print("\n=== Arena ===")
    print(f"{'model':<16}{'n':>7}{'logloss':>10}{'rps':>9}{'brier':>9}{'acc':>8}{'ece':>8}{'skill':>9}")
    rows = arena_table(conn, run_id)
    for row in rows:
        ll = row.get("log_loss")
        ll_s = f"{ll:>10.4f}" if isinstance(ll, float) else f"{'n/a':>10}"
        print(f"{row['model']:<16}{row.get('n', 0):>7}{ll_s}"
              f"{row.get('rps', float('nan')):>9.4f}{row.get('brier', float('nan')):>9.4f}"
              f"{row.get('accuracy', float('nan')):>8.4f}{row.get('ece', float('nan')):>8.4f}"
              f"{row.get('skill_vs_freq', float('nan')):>9.4f}")

    for model in models:
        if not getattr(model, "importances", None):
            continue
        print(f"\n=== {model.name} feature importance (top 15) ===")
        for fname, value in model.top_features(15):
            print(f"  {fname:<24}{value:>8.4f}")

    diagnostics = arena_diagnostics(conn, run_id)
    if diagnostics:
        print("\n=== Ensemble candidates (mean prior-block log loss) ===")
        print(f"{'stage':<12}{'method':<14}{'blocks':>8}{'mean logloss':>14}{'selected':>10}")
        for row in diagnostics:
            print(f"{row['stage']:<12}{row['method']:<14}{row['blocks']:>8}"
                  f"{row['mean_score']:>14.4f}{row['selected_blocks']:>10}")

    singles = [r for r in rows if r["model"] not in ("ensemble", "ensemble_cal")
               and isinstance(r.get("log_loss"), float) and not np.isnan(r["log_loss"])]
    ens = next((r for r in rows if r["model"] == "ensemble"), None)
    if ens and singles:
        best = min(singles, key=lambda r: r["log_loss"])
        cmp = compare_models(conn, run_id, "ensemble", best["model"])
        if cmp.get("n"):
            print(f"\nensemble vs {best['model']}: mean loss diff {cmp['mean_diff']:+.4f} "
                  f"95% CI [{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}] over "
                  f"{cmp['blocks']} blocks -> "
                  f"{'significant' if cmp['significant'] else 'not significant'}")
        cmp_mkt = compare_models(conn, run_id, "ensemble", "market_close")
        if cmp_mkt.get("n"):
            print(f"ensemble vs market_close: mean loss diff {cmp_mkt['mean_diff']:+.4f} "
                  f"95% CI [{cmp_mkt['ci_low']:+.4f}, {cmp_mkt['ci_high']:+.4f}] -> "
                  f"{'significant' if cmp_mkt['significant'] else 'not significant'}")

    if holdout_broken(cfg, conn):
        print("\n!! holdout_broken: the config changed after the headline run")
    conn.close()
    return 0


def cmd_holdout_declare(args) -> int:
    from .eval.holdout import season_windows, write_lock

    cfg = load_config(args.config)
    conn = open_db(cfg.db_path)
    win = season_windows(conn, cfg)
    p = write_lock(cfg, win.headline_start)
    print(f"holdout declared for headline window starting {win.headline_start}")
    print(f"config hash {cfg.config_hash}")
    print(f"written to {p}")
    conn.close()
    return 0


def cmd_predict(args) -> int:
    from .predict import run_live_predictions

    cfg = load_config(args.config)
    conn = open_db(cfg.db_path)
    try:
        report = run_live_predictions(cfg, conn, days=int(args.days))
        print(f"{report['fixtures']} upcoming fixture(s) across {len(report['leagues'])} league(s)")
        for league, entry in report["leagues"].items():
            print(f"  {league}: {entry['fixtures']} fixture(s), {entry['appended']} appended, "
                  f"{entry['skipped']} skipped, {entry['late']} late")
        print(f"ledger: {report['appended']} appended, {report['skipped']} skipped "
              f"(already predicted today), {report['late']} late (kick-off passed)")
        return 0
    finally:
        conn.close()


def cmd_ledger(args) -> int:
    from .ledger import verify

    cfg = load_config(args.config)
    conn = open_db(cfg.db_path)
    try:
        result = verify(conn)
        if result.get("ok"):
            print(f"chain intact: {result['checked']} row(s) verified")
            return 0
        print(f"chain broken at row {result.get('row_id')}: {result.get('reason')}")
        return 1
    finally:
        conn.close()


def cmd_serve(args) -> int:
    import uvicorn

    cfg = load_config(args.config)
    host = str(cfg.server.get("host", "127.0.0.1"))
    port = int(args.port or cfg.server.get("port", 8000))
    print(f"PRORRHESIS serving on http://{host}:{port}  (config hash {cfg.config_hash[:12]})")
    uvicorn.run("prorrhesis.api.app:app", host=host, port=port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m prorrhesis", description="PRORRHESIS match-prediction lab")
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="download + parse + upsert football-data.co.uk CSVs")
    ing.add_argument("--force", action="store_true", help="re-download even cached seasons")
    ing.add_argument("--discover", action="store_true", help="collect distinct club names without upserting")
    ing.set_defaults(func=cmd_ingest)

    names = sub.add_parser("names", help="canonical team names")
    nsub = names.add_subparsers(dest="names_command", required=True)
    sug = nsub.add_parser("suggest", help="seed canonical teams and suggest aliases")
    sug.set_defaults(func=cmd_names_suggest)

    ce = sub.add_parser("clubelo", help="optional ClubElo ratings")
    cesub = ce.add_subparsers(dest="clubelo_command", required=True)
    cef = cesub.add_parser("fetch", help="download ClubElo snapshots")
    cef.add_argument("--force", action="store_true", help="re-download even cached snapshots")
    cef.set_defaults(func=cmd_clubelo_fetch)

    serve = sub.add_parser("serve", help="start the local web app")
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=cmd_serve)

    bt = sub.add_parser("backtest", help="walk-forward evaluation")
    bt.add_argument("--kind", choices=["tuning", "headline"], default="tuning")
    bt.add_argument("--models", default=None, help="comma separated model names")
    bt.add_argument("--tune", action="store_true", help="tune hyperparameters on the burn-in first")
    bt.add_argument("--tune-dc", action="store_true",
                    help="search Dixon-Coles hyperparameters on the tuning window")
    bt.add_argument("--ensemble", action="store_true",
                    help="also build the ensemble (stacker + calibration)")
    bt.set_defaults(func=cmd_backtest)

    ho = sub.add_parser("holdout", help="frozen-config holdout lock")
    hosub = ho.add_subparsers(dest="holdout_command", required=True)
    dec = hosub.add_parser("declare", help="write holdout.lock for the current config")
    dec.set_defaults(func=cmd_holdout_declare)

    pred = sub.add_parser("predict", help="live predictions + ledger append")
    pred.add_argument("--days", type=int, default=7, help="how many days ahead to predict")
    pred.set_defaults(func=cmd_predict)

    led = sub.add_parser("ledger", help="the append-only prediction ledger")
    ledsub = led.add_subparsers(dest="ledger_command", required=True)
    ledv = ledsub.add_parser("verify", help="recompute the hash chain")
    ledv.set_defaults(func=cmd_ledger)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
