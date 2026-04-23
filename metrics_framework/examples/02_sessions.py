#!/usr/bin/env python
"""
Example 2: Sessions
===================

This example shows how to use sessions to group related metrics:
- Starting and ending sessions
- Recording within sessions
- Querying session data
- Using context managers
"""

from metrics_framework import MetricsEngine, MetricDefinition


def main():
    engine = MetricsEngine()

    # Define metrics
    engine.register_metric(MetricDefinition(name="action", metric_type="counter"))
    engine.register_metric(MetricDefinition(name="score", metric_type="gauge"))

    # --- Method 1: Manual session management ---
    print("=== Manual Session Management ===")

    session1 = engine.start_session(name="Game Round 1", user_id="player-42")
    print(f"Started session: {session1.session_id}")

    # Record actions during the session
    engine.record("action", 1, dimensions={"type": "jump"})
    engine.record("action", 1, dimensions={"type": "shoot"})
    engine.record("action", 1, dimensions={"type": "jump"})
    engine.record("score", 100)
    engine.record("score", 150)

    # End the session
    engine.end_session()
    print(f"Session duration: {session1.duration:.1f} seconds")
    print(f"Session summary: {session1.get_summary()}")

    # --- Method 2: Context manager ---
    print("\n=== Context Manager ===")

    with engine.start_session(name="Game Round 2", user_id="player-42") as session2:
        engine.record("action", 1, dimensions={"type": "run"})
        engine.record("action", 1, dimensions={"type": "shoot"})
        engine.record("score", 200)
        # Session automatically ends when exiting the block

    print(f"Session 2 duration: {session2.duration:.1f} seconds")

    # --- Query by session ---
    print("\n=== Query by Session ===")

    session1_data = engine.query(session_id=session1.session_id)
    session2_data = engine.query(session_id=session2.session_id)

    print(f"Session 1 records: {len(session1_data)}")
    print(f"Session 2 records: {len(session2_data)}")

    # --- Query by user across sessions ---
    print("\n=== Query by User ===")

    user_data = engine.query(user_id="player-42")
    print(f"Total records for player-42: {len(user_data)}")

    user_score = engine.aggregate("score", user_id="player-42")
    print(f"Average score for player-42: {user_score:.1f}")


if __name__ == "__main__":
    main()
