import logging
import asyncio

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import cartesia, openai, silero

logger = logging.getLogger("resume-agent")

# -----------------------------
# Intelligent Interrupt Control
# -----------------------------

agent_speaking = False
pending_interrupt = False

IGNORE_WORDS = ["yeah", "ok", "hmm", "uh-huh", "right"]
COMMAND_WORDS = ["stop", "wait", "cancel", "no"]

load_dotenv()

# This example shows how to resume an agent from a false interruption.
# If `resume_false_interruption` is True, the agent will first pause the audio output
# while not interrupting the speech before the `false_interruption_timeout` expires.
# If there is not new user input after the pause, the agent will resume the output for the same speech.
# If there is new user input, the agent will interrupt the speech immediately.

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=openai.STT(),
        tts=cartesia.TTS(),
        false_interruption_timeout=1.0,
        resume_false_interruption=True,
    )

    @session.on("audio_playback_started")
    def on_audio_start(event):
        global agent_speaking
        agent_speaking = True
        logger.info("Agent started speaking")


    @session.on("audio_playback_finished")
    def on_audio_end(event):
        global agent_speaking
        agent_speaking = False
        logger.info("Agent finished speaking")


    @session.on("interruption")
    def on_interrupt(event):
        global pending_interrupt

        if agent_speaking:
            pending_interrupt = True
            logger.info("Interrupt detected — waiting for transcript validation")



    @session.on("user_speech_final")
    def on_user_speech(event):

            async def process():
                global pending_interrupt

                user_text = event.text.lower().strip()
                words = user_text.split()

                logger.info(f"User said: {user_text}")

                # Agent speaking + interrupt pending
                if agent_speaking and pending_interrupt:

                    # Hard command interrupt
                    if any(cmd in user_text for cmd in COMMAND_WORDS):
                        await session.stop_audio()
                        pending_interrupt = False
                        logger.info("Command interruption accepted")
                        return

                    # Ignore fillers
                    elif all(word in IGNORE_WORDS for word in words):
                        pending_interrupt = False
                        logger.info("Passive acknowledgement ignored")
                        return

                    # Unknown input = interrupt
                    else:
                        await session.stop_audio()
                        pending_interrupt = False
                        logger.info("Unknown input treated as interruption")
                        return

                # Normal conversation
                if not agent_speaking:
                    await session.send_user_message(user_text)

                # IMPORTANT: Must be inside handler
                asyncio.create_task(process())


    await session.start(agent=Agent(instructions="You are a helpful assistant."), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
