"""Requeue dead-lettered jobs after the root cause is fixed:  python -m priorauth.scripts.replay_dead [--kind process_request]"""
import argparse

from priorauth.bootstrap import make_store

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--kind")
    print(f"requeued {make_store().requeue_dead(ap.parse_args().kind)} dead job(s)")
