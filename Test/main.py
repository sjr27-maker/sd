import argparse
from tutor import run_session
from teacher.report_generator import generate_report

def main():
    parser = argparse.ArgumentParser(description="SYRA Adaptive AI Tutor")
    parser.add_argument("--mode",       default="tutor",
                        choices=["tutor", "report"])
    parser.add_argument("--subject",    default="Mathematics")
    parser.add_argument("--grade",      type=int, default=9)
    parser.add_argument("--student",    default="student_001")
    args = parser.parse_args()

    if args.mode == "tutor":
        run_session(
            subject=args.subject,
            grade=args.grade,
            student_id=args.student
        )
    elif args.mode == "report":
        generate_report(args.student)

if __name__ == "__main__":
    main()