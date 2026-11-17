"""Run simulator-free contracts for the current attention selector and SF."""
import unittest

def main():
    suite=unittest.defaultTestLoader.loadTestsFromNames([
        "tsc_repaired.test_task_selector", "tsc_repaired.test_task_sf",
        "tsc_repaired.test_traffic_runtime", "paper_aligned_followup.test_evaluate"])
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1

if __name__ == "__main__": raise SystemExit(main())
