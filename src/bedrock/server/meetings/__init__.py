"""Recording a meeting in the browser and turning it into searchable notes.

This is the in-house half of the meeting story. `src/bedrock/server/tools/meeting_transcripts.py`
reads meetings that Zoom recorded for us; this package produces meetings that
nobody else recorded at all — someone in a room presses Record, or captures the
audio of an online call they are already sitting in.

Both ends meet in the same place. A recording made here is delivered to the
customer's datacenter connector in exactly the shape zoom-transcript-vault
delivers a Zoom transcript, so `meeting_transcripts` finds it with no new
retrieval code and "what did we decide on Tuesday?" works across both.

Nothing leaves the machine. Speech becomes text through whisper.cpp running
locally, and the notes are written by whichever local model the deployment is
already using. See `pipeline.py`.
"""
