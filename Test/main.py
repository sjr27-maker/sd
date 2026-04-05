import argparse
import asyncio
from tutor import run_session
from engine.live_session import run_live_session
from teacher.report_generator import generate_report
import pygame
# This must happen before anything else imports tts_client
pygame.mixer.pre_init(frequency=24000, size=-16, channels=2, buffer=2048)
pygame.mixer.init()

def main():
    parser = argparse.ArgumentParser(description="SYRA Adaptive AI Tutor")
    parser.add_argument("--mode",    default="tutor",
                        choices=["tutor", "live", "report"])
    parser.add_argument("--subject", default="Mathematics")
    parser.add_argument("--grade",   type=int, default=9)
    parser.add_argument("--student", default="student_001")
    args = parser.parse_args()

    if args.mode == "tutor":
        # Half-duplex — stable, tested
        run_session(args.subject, args.grade, args.student)

    elif args.mode == "live":
        # Full duplex — Gemini 3.1 Flash Live
        asyncio.run(run_live_session(
            args.subject, args.grade, args.student
        ))

    elif args.mode == "report":
        generate_report(args.student)

if __name__ == "__main__":
        main()